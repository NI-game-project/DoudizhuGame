import numpy as np
import torch
from utils_global import remove_illegal
from doudizhu.utils import opt_legal
from agents.networks import RainbowDQN
from agents.buffers import PrioritizedBufferWithSumTree


class RainbowAgent:
    """
    An implementation of rainbow dqn agent
    with double, dueling, noisy, c51(categorical/distribution) network, prioritized replay buffer,
    without multi-step
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 num_atoms=51,
                 v_min=-1.,
                 # max_reward of the game
                 v_max=1.,
                 lr=0.00001,
                 gamma=0.99,
                 epsilons=None,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(2e4),
                 replay_memory_init_size=1000,
                 soft_update=True,
                 # for soft_update
                 soft_update_target_every=10,
                 # for hard_update
                 hard_update_target_every=1000,
                 use_conv=False,
                 device=None, ):

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.use_raw = False
        self.state_shape = state_shape  # (8,5,15)
        self.num_actions = num_actions  # 309
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.lr = lr
        self.gamma = gamma
        self.soft_update = soft_update
        self.soft_update_every = soft_update_target_every
        self.hard_update_every = hard_update_target_every
        self.tau = 1e-3
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)
        self.epsilon = 0

        self.batch_size = batch_size
        self.train_every = train_every
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.device = device
        self.use_conv = use_conv
        self.use_raw = False

        # Total time steps
        self.timestep = 0

        # initialize q and target networks
        self.local_net = RainbowDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                    num_atoms=self.num_atoms, v_min=self.v_min, v_max=self.v_max,
                                    use_conv=self.use_conv).to(self.device)
        self.target_net = RainbowDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                     num_atoms=self.num_atoms, v_min=self.v_min, v_max=self.v_max,
                                     use_conv=self.use_conv).to(self.device)

        self.local_net.eval()
        self.target_net.eval()

        # initialize optimizer(Adam) for q network
        self.optim = torch.optim.Adam(self.local_net.parameters(), lr=lr)

        self.softmax = torch.nn.Softmax(dim=-1)

        # initialize memory buffer
        self.memory_buffer = PrioritizedBufferWithSumTree(replay_memory_size, batch_size)

        # for logging
        self.loss = 0
        self.actions = []
        self.predictions = []
        self.q_values = 0
        self.current_q_values = 0
        self.expected_q_values = 0

    @staticmethod
    def KL_divergence_two_dist(dist_p, dist_q):
        kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
        return kld

    def projection_distribution(self, local_net, target_net, next_states, rewards, dones):
        # support: [num_atoms]
        # [-1., -0.96, -0.92, ..., 0.92, 0.96, 1.]
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        # delta_z: scalar - 0.04
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        with torch.no_grad():
            # next_dist: [batch_size, num_actions, num_atoms]
            target_next_dist = target_net(next_states)
            next_dist = local_net(next_states)
        # next_action: [batch_size]
        next_q_values = (next_dist * support).sum(2)
        next_action = next_q_values.max(1)[1]

        # next_action: [batch_size, 1, num_atoms]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)

        # next_dist: [batch_size, num_atoms]
        next_dist = target_next_dist.gather(1, next_action).squeeze(1)

        # [batch_size, num_atoms]
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        # Compute projection of the application of the Bellman operator.
        # compute projected values for each particular atom zi according to: zi=r+γzi
        # and clip the obtained value to [Vmin, Vmax]
        Tz = rewards + (1 - dones) * support * self.gamma
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)

        # Compute categorical indices for distributing the probability
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # [batch_size, num_atoms]
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long() \
            .unsqueeze(1).expand(self.batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        # The projected atom zi lies between two “real” atoms zl and zu, with a non-integer index b
        # (for example b=3.4, l=3 and u=4).
        # The corresponding probability pb(s′,a′;θ) of the next greedy action (s′,a′) is “spread” to its neighbors
        # through a local interpolation depending on the distances between b, l and u:
        # Δpl(s′,a′;θ)=pb(s′,a′;θ)(b−u)
        # Δpu(s′,a′;θ)=pb(s′,a′;θ)(l−b)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def predict(self, state):
        """
        predict an action given state
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            max_action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions
            probs (np.array) : softmax distribution over legal actions
            predicted_action(int): integer of action id, argmax_action predicted by the local_network

        """
        state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
        legal_actions = opt_legal(state['legal_actions'])
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        with torch.no_grad():
            # Distribution of the probabilities of θ(s,a) on the support
            dist = self.local_net(state_obs).cpu().detach()
            dist = dist.mul(support).numpy()
            q_values = dist.sum(2)[0]

            predicted_action = np.argmax(q_values)
            probs = remove_illegal(q_values, legal_actions)
            max_action = np.argmax(probs)

            self.q_values = q_values

        return probs, max_action, predicted_action

    def step(self, state):
        """
            Pick an action given a state using epsilon greedy action selection
            Input:
                state (dict)
                    'obs' : actual state representation
                   'legal_actions' : possible legal actions to be taken from this state
            Output:
                action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions

        """

        self.epsilon = self.epsilons[min(self.timestep, self.epsilon_decay_steps - 1)]
        legal_actions = opt_legal(state['legal_actions'])
        max_action = self.predict(state)[1]
        # pick an action randomly from optimized legal_actions
        if np.random.uniform() < self.epsilon:
            probs = remove_illegal(np.ones(self.num_actions), legal_actions)
            action = np.random.choice(self.num_actions, p=probs)
        # pick the argmax_action predicted by the network
        else:
            action = max_action

        return action

    def eval_step(self, state, use_max=True):
        """
        This is to be used during evaluation,
        Pick an action given a state according to max q value, so no epsilon greedy when actually selecting the action.
        Calling the predict function to produce the action
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
            use_max (bool) : should we return best action or select according to distribution
        Output:
            max_action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions
            probs (np.array) : softmax distribution over legal actions
        """

        probs, max_action, predicted_action = self.predict(state)
        if use_max:
            action = max_action
        else:
            action = np.random.choice(self.num_actions, p=probs)

        self.actions.append(max_action)
        self.predictions.append(predicted_action)

        return action, probs

    def add_transition(self, transition):
        """
        Add transition to memory buffer and train the network one batch.
        input:
            transition (tuple): tuple representation of a transition --> (state, action, reward, next_state, done)
        output:
            Nothing. Stores transition in the buffer, and updates the network using memory buffer, and soft/hard updates
            the target network depending on timesteps.
        """
        state, action, reward, next_state, done = transition

        # store transition in memory
        self.memory_buffer.save(state['obs'], action, reward, next_state['obs'], done)

        self.timestep += 1

        # once we have enough samples, get a sample from stored memory to train the network
        if self.timestep >= self.replay_memory_init_size and \
                self.timestep % self.train_every == 0:
            batch_loss = self.train()
            print(f'\rstep: {self.timestep}, loss on batch: {batch_loss}', end='')

        # soft/hard update the parameters of the target network
        if self.soft_update:
            # target_weights = target_weights * (1-tau) + q_weights * tau, where 0 < tau < 1
            if self.timestep >= self.replay_memory_init_size and self.timestep % self.soft_update_every == 0:
                for target_param, local_param in zip(self.target_net.parameters(), self.local_net.parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
                # print(f'target parameters soft_updated on step {self.timestep}')
        else:
            if self.timestep >= self.replay_memory_init_size and self.timestep % self.hard_update_every == 0:
                self.target_net.load_state_dict(self.local_net.state_dict())
                # print(f'target parameters hard_updated on step {self.timestep}')

        self.target_net.eval()

    def train(self):
        """
        Sample from memory buffer and train the network one step.
        Input:
            Nothing. Draws sample from memory buffer to train the network
        Output:
            loss (float) : loss on training batch
        """
        states, actions, rewards, next_states, dones, indices, is_weights = self.memory_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # Compute distribution of Q(s',a)
        proj_dists = self.projection_distribution(self.local_net, self.target_net, next_states, rewards, dones)

        # Compute probabilities of Q(s,a*)
        dists = self.local_net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dists = dists.gather(1, actions).squeeze(1)
        # trick for avoiding nans
        dists.detach().data.clamp_(0.01, 0.99)

        # minimizing the KL divergence is the same as minimizing the cross-entropy between the two
        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        error = -(proj_dists * dists.log()).sum(1).unsqueeze(1) * is_weights
        loss = error.mean()

        # update per priorities
        self.memory_buffer.update_priorities(indices, abs(error.detach().numpy()))

        self.expected_q_values = proj_dists
        self.current_q_values = dists

        self.optim.zero_grad()
        self.local_net.train()
        loss.backward()

        # Clip gradients (normalising by max value of gradient L2 norm)
        # nn.utils.clip_grad_norm_(self.local_net.parameters(), max_norm=2)
        # nn.utils.clip_grad_value_(self.local_net.parameters(), clip_value=0.5)
        self.optim.step()

        self.local_net.reset_noise()
        self.target_net.reset_noise()
        self.local_net.eval()

        self.loss = loss.item()
        self.actions = actions

        return loss.item()

    def save_state_dict(self, file_path):
        """
        save state dict for the networks of DQN agent
        Input:
            file_path (str): string filepath to save the agent at
        """

        state_dict = dict()
        state_dict['local_net'] = self.local_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
        Load agent parameters from filepath
        Input:
            file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.local_net.load_state_dict(state_dict['local_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
