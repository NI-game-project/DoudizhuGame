import numpy as np
import torch
import torch.nn as nn
from agents.common.model import DQN, DeepConvNet
from utils_global import remove_illegal, action_mask
from agents.common.buffers import BasicBuffer

"""
An implementation of dqn Agent
"""


class DQNBaseAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        lr (float) : learning rate to use for training online_net
        gamma (float) : discount parameter
        epsilon_start (float) : start value of epsilon
        epsilon_end (float) : stop value of epsilon
        epsilon_decay_steps (int) : how often should we decay epsilon value
        batch_size (int) : batch sizes to use when training networks
        train_every (int) : how often to update the online work
        replay_memory_init_size (int) : minimum number of experiences to start training
        replay_memory_size (int) : max number of experiences to store in memory buffer
        soft_update (bool) : if update the target network softly or hardly
        soft_update_target_every (int) : how often to soft update the target network
        hard_update_target_every (int): how often to hard update the target network(copy the param of online network)
        loss_type (str) : which loss to use, ('mse' / 'huber')
        noisy (bool) : True if use NoisyLinear for network False Linear
        clip (bool) : if gradient is clipped(norm / value)
        use_conv (bool) : if use convolutional layers for network
        #deep_conv (bool) : if use deep convolutional layers for network
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 lr=0.0001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(2e5),
                 replay_memory_init_size=1000,
                 soft_update=False,
                 soft_update_target_every=10,
                 hard_update_target_every=1000,
                 loss_type='huber',
                 double=False,
                 noisy=False,
                 clip=True,
                 use_conv=False,
                 device=None,
                 ):

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.use_raw = False

        self.state_shape = state_shape  # (n,4,15)
        self.num_actions = num_actions  # 309

        self.lr = lr
        self.gamma = gamma
        self.soft_update = soft_update
        self.soft_update_every = soft_update_target_every
        self.tau = 1e-3
        self.hard_update_every = hard_update_target_every
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon = 0
        self.batch_size = batch_size
        self.train_every = train_every
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size

        self.noisy = noisy
        self.double = double
        self.clip = clip
        self.clip_norm = 2
        self.clip_value = 0.5
        self.use_conv = use_conv

        # Total time steps and training time steps
        self.total_time = 0
        self.train_time = 0

        # initialize online and target networks
        # self.online_net = DeepConvNet(state_shape=self.state_shape, action_num=self.num_actions, kernels=64)
        # self.target_net = DeepConvNet(state_shape=self.state_shape, action_num=self.num_actions, kernels=64)

        self.online_net = DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                              use_conv=self.use_conv, noisy=self.noisy).to(self.device)
        self.target_net = DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                              use_conv=self.use_conv, noisy=self.noisy).to(self.device)

        self.online_net.train()
        self.target_net.train()
        # Disable calculations of gradients of the target network.
        for param in self.target_net.parameters():
            param.requires_grad = False

        # initialize optimizer for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)

        # initialize loss function for network
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss(reduction='mean')

        self.softmax = torch.nn.Softmax(dim=-1)

        # initialize memory buffer
        self.memory_buffer = BasicBuffer(replay_memory_size, batch_size)

        # for plotting
        self.loss = 0
        self.actions = []
        self.predictions = []
        self.q_values = 0
        self.current_q_values = 0
        self.expected_q_values = 0

    def predict(self, state):
        """
        predict an action given state (with/without noisy weights)
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            max_action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions
            probs (np.array) : softmax distribution of q_values over legal actions
            predicted_action(int): integer of action id, argmax_action predicted by the local_network

        """

        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
            legal_actions = state['legal_actions']
            q_values = self.online_net(state_obs)[0].cpu().detach().numpy()
            # Note the first arg for remove_illegal func must be probabilities(i.e., elements must be larger than zero)
            # Calculate a softmax distribution over q_values to ensure probs passed to remove_illegal larger than zero
            softmax_q_vals = self.softmax(self.online_net(state_obs))[0].cpu().detach().numpy()
            predicted_action = np.argmax(softmax_q_vals)
            probs = remove_illegal(softmax_q_vals, legal_actions)
            max_action = np.argmax(probs)

            self.q_values = q_values

        return probs, max_action, predicted_action

    def step(self, state):
        """
            Pick an action given a state using epsilon greedy action selection
             Calling the predict function to produce the action
            Input:
                state (dict)
                    'obs' : actual state representation
                    'legal_actions' : possible legal actions to be taken from this state
            Output:
                action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions

        """
        if self.noisy:
            # if use noisy layer for network, epsilon greedy is not needed, set epsilon to 0
            self.epsilon = 0
            self.reset_noise()
        else:
            self.epsilon = self.epsilons[min(self.total_time, self.epsilon_decay_steps - 1)]

        legal_actions = state['legal_actions']
        max_action = self.predict(state)[1]
        # pick an action randomly
        if np.random.uniform() < self.epsilon:
            probs = remove_illegal(np.ones(self.num_actions), legal_actions)
            action = np.random.choice(self.num_actions, p=probs)
        # pick the argmax_action predicted by the network
        else:
            action = max_action

        return action

    def eval_step(self, state, use_max=True):
        """
        This is to be used during evaluation.
        Pick an action given a state according to max q_value, no epsilon greedy needed when selecting the action.
        Calling the predict function to produce the action
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
            use_max (bool) : should we return best action or select according to distribution
        Output:
            action (int) : action id, argmax_action predicted by the local_network after removing illegal_actions
            probs (np.array) : softmax distribution over legal actions
        """
        # Set online network to evaluation mode
        self.eval_mode()

        probs, max_action, predicted_action = self.predict(state)
        if use_max:
            action = max_action
        else:
            action = np.random.choice(self.num_actions, p=probs)

        self.actions.append(max_action)
        self.predictions.append(predicted_action)

        self.train_mode()

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
        self.memory_buffer.save(state['obs'], state['legal_actions'], action, reward,
                                next_state['obs'], next_state['legal_actions'], done)

        self.total_time += 1

        # once we have enough samples, get a sample from stored memory to train the network
        if self.total_time >= self.replay_memory_init_size and \
                self.total_time % self.train_every == 0:

            self.train_mode()
            if self.noisy:
                # Sample a new set of noisy epsilons,
                # (i.e. fix new random weights for noisy layers to encourage exploration)
                self.online_net.reset_noise()
                self.target_net.reset_noise()

            batch_loss = self.train()
            print(f'\rstep: {self.total_time}, loss on batch: {batch_loss}', end='')

    def train(self):
        """
        Sample from memory buffer and train the network one step.
        Input:
            Nothing. Draws sample from memory buffer to train the network
        Output:
            loss (float) : loss on training batch
        """
        # sample a batch of transitions from memory
        states, legal_actions, actions, rewards, next_states, next_legal_actions, dones = self.memory_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        self.online_net.train()
        self.target_net.train()

        # Calculate q values of current (states, actions).
        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Calculate q values of next states.
            next_q_values_target = self.target_net(next_states)

            # Do action mask for q_values of next_state if not done (i.e., set q_values of illegal actions to -inf)
            for i in range(self.batch_size):
                next_q_values_target[i] = action_mask(self.num_actions, next_q_values_target[i], next_legal_actions[i])

            # Select greedy actions a∗ in next state using the target(online if double) network., a∗=argmaxa′Qθ(s′,a′)
            next_argmax_actions = next_q_values_target.max(1)[1]
            # Use greedy actions to select target q values.
            next_q_values = next_q_values_target.gather(1, next_argmax_actions.unsqueeze(1)).squeeze(1)

        # Compute the expected q value y=r+γQθ′(s′,a∗)
        # for double dqn:
        # value = reward + gamma * target_network.predict(next_state)[argmax(local_network.predict(next_state))]
        expected_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        expected_q_values.detach()

        loss = self.criterion(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients (normalising by max value of gradient L2 norm)
        if self.clip:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.clip_norm)
            # nn.utils.clip_grad_value_(self.online_net.parameters(), clip_value=0.5)
        self.optimizer.step()

        self.loss = loss.item()

        # soft/hard update the parameters of the target network and increase the training time
        self.update_target_net(self.soft_update)
        self.train_time += 1

        self.expected_q_values = expected_q_values
        self.current_q_values = q_values

        return loss.item()

    def update_target_net(self, is_soft):
        """Updates target network """

        if is_soft:
            # target_weights = target_weights * (1-tau) + online_weights * tau,(0<tau<1)
            if self.train_time > 0 and self.train_time % self.soft_update_every == 0:
                for target_param, local_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
                # print(f'target parameters soft_updated on step {self.train_time}')

        else:
            # copy the parameters of online network to target network
            if self.train_time > 0 and self.train_time % self.hard_update_every == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                # print(f'target parameters hard_updated on step {self.train_time}')

    def reset_noise(self):
        """Resets noisy weights in all linear layers (of online net only) """
        self.online_net.reset_noise()

    def train_mode(self):
        """set the online network to training mode"""
        self.online_net.train()

    def eval_mode(self):
        """set the online network to evaluation mode"""
        self.online_net.eval()

    def save_state_dict(self, file_path):
        """
        save state dict for the online and target networks of agent
        Input:
            file_path (str): string filepath to save the agent at
        """

        state_dict = dict()
        state_dict['online_net'] = self.online_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
        Load agent's parameters from filepath
        Input:
            file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(state_dict['online_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
