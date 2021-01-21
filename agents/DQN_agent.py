import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from agents.networks import DQNet
from utils_global import remove_illegal

# adding a small penalty if the network predicts the legal actions but not pass

class DQNAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        hidden_layers (list) : hidden layer size
        lr (float) : learning rate to use for training q_net
        batch_size (int) : batch sizes to use when training networks
        replay_memory_size (int) : max number of experiences to store in memory buffer
        update_target_every (int) : how often to update parameters of the target network
        epsilon_decay_steps (int) : how often should we decay epsilon value
        gamma (float) : discount parameter
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 hidden_layers,
                 lr=0.0001,
                 gamma=0.99,
                 epsilons=None,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=20000,
                 replay_memory_init_size=1000,
                 update_target_every=1000,
                 device=None, ):

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.use_raw = False
        self.state_shape = state_shape # (6,5,15)
        self.num_actions = num_actions # 309
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        # self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)
        # for dqn_rule_rule_2nd
        self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)

        self.batch_size = batch_size
        self.train_every = train_every
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_every = update_target_every
        self.device = device
        self.use_raw = False

        # Total time steps
        self.timestep = 0

        # initialize q and target networks
        self.q_net = DQNet(state_shape=state_shape,
                           num_actions=num_actions,
                           mlp_layers=hidden_layers).to(device)
        self.target_net = DQNet(state_shape=state_shape,
                                num_actions=num_actions,
                                mlp_layers=hidden_layers).to(device)
        self.q_net.eval()
        self.target_net.eval()

        # initialize optimizer(Adam) for q network
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # initialize loss func(mse_loss) for network
        self.loss = nn.MSELoss(reduction='mean')

        self.softmax = torch.nn.Softmax(dim=1)

        # initialize memory buffer
        self.memory_buffer = Memory(replay_memory_size, batch_size)

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

        epsilon = self.epsilons[min(self.timestep, self.epsilon_decay_steps - 1)]
        legal_actions = state['legal_actions']
        max_action = self.predict(state)[1]
        if np.random.uniform() < epsilon:
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
        probs, max_action = self.predict(state)
        if use_max:
            action = max_action
        else:
            action = np.random.choice(self.num_actions, p=probs)

        return action, probs

    def predict(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            legal_actions = state['legal_actions']
            # calculate a softmax distribution over the q_values for all actions
            softmax_q_vals = self.softmax(self.q_net(state_obs))[0].cpu().detach().numpy()
            predicted_action = np.argmax(softmax_q_vals)
            probs = remove_illegal(softmax_q_vals, legal_actions)
            max_action = np.argmax(probs)

        return probs, max_action

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
        self.memory_buffer.save(state['obs'], state['legal_actions'], action, reward, next_state['obs'], done)

        self.timestep += 1

        # once we have enough samples, get a sample from stored memory to train the network
        if self.timestep >= self.replay_memory_init_size and \
                self.timestep % self.train_every == 0:
            batch_loss = self.train()
            print(f'\rstep: {self.timestep}, loss on batch: {batch_loss}', end='')

        # update the parameters of the target network
        if self.timestep % self.update_target_every == 0:
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
        states, legal_actions, actions, rewards, next_states, dones = self.memory_buffer.sample()

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = (1 - torch.FloatTensor(dones)).to(self.device)

        penalty = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if actions[i] in legal_actions[i] and actions[i] != self.num_actions - 1:
                penalty[i] += 0.2
        penalty = torch.FloatTensor(penalty).to(self.device)

        q_values = self.q_net(states)
        next_q_values = self.target_net(next_states)
        argmax_actions = self.q_net(next_states).max(1)[1].detach()
        next_q_values = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_q_values = rewards + self.gamma * dones * next_q_values + penalty
        expected_q_values.detach()

        loss = self.loss(q_values, expected_q_values)

        self.q_net.train()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.q_net.eval()
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


Transition = namedtuple('Transition', ['state', 'legal_actions', 'action', 'reward', 'next_state', 'done'])


class Memory(object):
    """
    Memory for saving transitions
    """

    def __init__(self, memory_size, batch_size):
        """
            Initialize
            Args:
            memory_size (int): the size of the memory buffer
        """

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, legal_actions, action, reward, next_state, done):
        """
            Save transition into memory

            Args:
                state (numpy.array): the current state
                legal_actions (list): a list of legal actions
                action (int): the performed action ID
                reward (float): the reward received
                next_state (numpy.array): the next state after performing the action
                done (boolean): whether the episode is finished
        """

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, legal_actions, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        """
            Sample a minibatch from the replay memory

            Returns:
                state_batch (list): a batch of states
                legal_actions(list): a batch of legal_actions
                action_batch (list): a batch of actions
                reward_batch (list): a batch of rewards
                next_state_batch (list): a batch of states
                done_batch (list): a batch of dones
        """

        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
