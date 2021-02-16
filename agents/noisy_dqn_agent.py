import numpy as np
import torch
import torch.nn as nn
from agents.networks import DQNNoisy
from utils_global import remove_illegal
from agents.buffers import PrioritizedReplayBuffer


class NoisyDQNAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        lr (float) : learning rate to use for training q_net
        batch_size (int) : batch sizes to use when training networks
        replay_memory_size (int) : max number of experiences to store in memory buffer
        epsilon_decay_steps (int) : how often should we decay epsilon value
        gamma (float) : discount parameter
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 lr=0.00001,
                 gamma=0.97,
                 epsilons=None,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(1e5),
                 replay_memory_init_size=100,
                 soft_update=True,
                 # for soft_update
                 soft_update_target_every=10,
                 # for hard_update
                 hard_update_target_every=1000,
                 device=None, ):

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.use_raw = False
        self.state_shape = state_shape  # (8,5,15)
        self.num_actions = num_actions  # 309
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
        self.use_raw = False

        # Total time steps
        self.timestep = 0

        # initialize q and target networks
        self.q_net = DQNNoisy(state_shape=state_shape, num_actions=num_actions).to(device)
        self.target_net = DQNNoisy(state_shape=state_shape, num_actions=num_actions).to(device)

        self.q_net.eval()
        self.target_net.eval()

        # initialize optimizer(Adam) for q network
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # initialize loss func(mse_loss) for network
        # self.criterion = nn.MSELoss(reduction='mean')
        self.criterion = nn.SmoothL1Loss(reduction='mean')

        self.softmax = torch.nn.Softmax(dim=-1)

        # initialize memory buffer
        self.memory_buffer = PrioritizedReplayBuffer(replay_memory_size, batch_size)

        # for plotting
        self.loss = 0
        self.actions = []
        self.predictions = []
        self.q_values = 0
        self.current_q_values = 0
        self.expected_q_values = 0

    def step(self, state):
        """
            Given state, produce actions to generate training data. Use epsilon greedy action selection.
            Should be separate from compute graph as we only update through the feed function.
            Uses epsilon greedy methods in order to produce the action.
            Pick an action given a state using epsilon greedy action selection
            Input:
                state (dict)
                    'obs' : actual state representation
                    'legal_actions' : possible legal actions to be taken from this state
            Output:
                action (int) : integer representing action id
        """

        self.epsilon = self.epsilons[min(self.timestep, self.epsilon_decay_steps - 1)]
        legal_actions = state['legal_actions']
        max_action = self.predict(state)[1]
        if np.random.uniform() < self.epsilon:
            probs = remove_illegal(np.ones(self.num_actions), legal_actions)
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = max_action

        return action

    def eval_step(self, state, use_max=True):
        """
        Pick an action given a state according to max q value. This is to be used during evaluation,
        so no epsilon greedy when actually selecting the action. calling the predict function to produce the action
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
            use_max (bool) : should we return best action or select according to distribution
        Output:
            action (int) : integer representing action id
            probs (np.array) : softmax distribution over the actions
        """
        probs, max_action, predicted_action = self.predict(state)
        if use_max:
            action = max_action
        else:
            action = np.random.choice(self.num_actions, p=probs)

        self.actions.append(max_action)
        self.predictions.append(predicted_action)

        return action, probs

    def predict(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
            legal_actions = state['legal_actions']
            q_values = self.q_net(state_obs)[0].cpu().detach().numpy()

            #predicted_action = np.argmax(q_values)
            #probs = remove_illegal(q_values, legal_actions)
            # calculate a softmax distribution over the q_values for all actions
            softmax_q_vals = self.softmax(self.q_net(state_obs))[0].cpu().detach().numpy()
            predicted_action = np.argmax(softmax_q_vals)
            probs = remove_illegal(softmax_q_vals, legal_actions)

            max_action = np.argmax(probs)

            self.q_values = q_values

        return probs, max_action, predicted_action

    def add_transition(self, transition):
        """
        Add transition to memory buffer and train the network one batch.
        input:
            transition (tuple): tuple representation of a transition --> (state, action, reward, next_state, done)
        output:
            Nothing. Stores transition in the buffer, and updates the network using memory buffer, and updates the
            target network depending on timesteps.
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

        # update the parameters of the target network
        if self.soft_update:
            if self.timestep % self.soft_update_every == 0:
                for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                    target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        else:
            if self.timestep % self.hard_update_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            self.target_net.eval()
            # print(f'target parameters updated on step {self.timestep}')

    def train(self):
        """
        Samples from memory buffer and trains the network one step.
        Input:
            Nothing. Draws sample from memory buffer to train the network
        Output:
            loss (float) : loss on training batch
        """
        states, actions, rewards, next_states, dones, indices, weights = self.memory_buffer.sample()

        self.optim.zero_grad()
        self.q_net.train()

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = (1 - torch.FloatTensor(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            next_q_state_values = self.target_net(next_states)
            next_argmax_actions = next_q_values.max(1)[1]
            next_q_values = next_q_state_values.gather(1, next_argmax_actions.unsqueeze(1)).squeeze(1)

        expected_q_values = rewards + self.gamma * dones * next_q_values
        expected_q_values.detach()
        q_values = self.q_net(states)
        prediction = q_values.max(1)[1]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        self.current_q_values = q_values
        self.expected_q_values = expected_q_values

        loss = (expected_q_values - q_values).pow(2) * weights
        priorities = loss + 1e-5
        priorities = priorities.detach().numpy()
        self.memory_buffer.update_priorities(indices, priorities)
        loss = loss.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=2)
        # nn.utils.clip_grad_value_(self.q_net.parameters(), clip_value=0.5)
        self.optim.step()
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        self.q_net.eval()

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
        state_dict['q_net'] = self.q_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
        Load agent parameters from filepath
        Input:
            file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state_dict['q_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
