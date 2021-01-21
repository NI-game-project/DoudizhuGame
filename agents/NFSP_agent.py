import numpy as np
import torch

from agents.networks import DQNet, AveragePolicyNet
from agents.DQN_agent_v5 import Memory
from utils_global import remove_illegal
from random import sample
from torch import nn

# adding a penalty to rl_agent
# sl_loss: optimizing the log-prob of past actions taken   L = E(-log(ap_net(s,a)))


class NFSPAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        sl_hidden_layers (list) : hidden layer sizes to use for average policy net for supervised learning
        rl_hidden_layers (list) : hidden layer sizes to use for best response net for reinforcement learning
        sl_lr (float) : learning rate to use for training average policy net
        rl_lr (float) : learning rate to use for training action value net
        batch_size (int) : batch sizes to use when training networks
        rl_memory_size (int) : max number of experiences to store in reinforcement learning memory buffer
        sl_memory_size (int) : max number of experiences to store in supervised learning memory buffer
        q_update_every (int) : how often to copy parameters to target network
        epsilons (list) : list of epsilon values to use over training period
        epsilon_decay_steps (int) : how often should we decay epsilon value
        eta (float) : anticipatory parameter for NFSP
        gamma (float) : discount parameter
        device (torch.device) : device to put models on
    """
    def __init__(self,
                 # these are the parameters in nfsp paper
                 scope,
                 num_actions,
                 state_shape,
                 sl_hidden_layers,
                 rl_hidden_layers,
                 sl_lr=.005,
                 rl_lr=.1,
                 batch_size=256,
                 train_every=128,
                 rl_memory_init_size=1000,
                 rl_memory_size=int(2e5),
                 sl_memory_init_size=1000,
                 sl_memory_size=int(2e6),
                 q_update_every=1000,
                 q_train_every=128,
                 epsilons=None,
                 epsilon_decay_steps=int(1e5),
                 eta=.2,
                 gamma=.99,
                 device=None):
        self.scope = scope
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.update_every = q_update_every
        self.train_every = train_every
        self.discount_factor = gamma
        self.sl_memory_init_size = sl_memory_init_size
        self.rl_memory_init_size = rl_memory_init_size
        self.q_train_every = q_train_every
        self.anticipatory_param = eta
        self.device = device
        self.use_raw = False

        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(0.08, 0.0, epsilon_decay_steps)
        # self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)

        # average policy can be modeled as a Deep Q Network and we take softmax after final layer
        self.average_policy = AveragePolicyNet(state_shape=state_shape,
                                               num_actions=num_actions,
                                               mlp_layers=sl_hidden_layers,
                                               activation='relu').to(self.device)
        self.average_policy.eval()

        # action value and target network are Deep Q Networks
        self.dqn_eval = DQNet(state_shape=state_shape,
                              num_actions=num_actions,
                              mlp_layers=rl_hidden_layers,
                              activation='relu').to(device)
        self.dqn_target = DQNet(state_shape=state_shape,
                                num_actions=num_actions,
                                mlp_layers=rl_hidden_layers,
                                activation='relu').to(device)
        self.dqn_eval.eval()
        self.dqn_target.eval()

        # initialize loss functions

        self.rl_loss = nn.MSELoss(reduction='mean')

        # initialize optimizers
        """
        in the paper: using sgd optim, eta = 0.1.
        rl_lr = 0.1, sl_lr = 0.005,
        epsilon decay from 0.06 to 0
        """
        self.sl_optim = torch.optim.Adam(self.average_policy.parameters(), lr=sl_lr)
        self.rl_optim = torch.optim.Adam(self.dqn_eval.parameters(), lr=rl_lr)

        # initialize memory buffers
        self.rl_buffer = Memory(rl_memory_size, batch_size)
        self.sl_buffer = ReservoirMemoryBuffer(sl_memory_size, batch_size)

        # current policy
        self.policy = None
        self.eval_policy = 'average policy'

        self.softmax = torch.nn.Softmax(dim=1)

        self.timestep = 0

    def set_policy(self, policy=None):
        """
            Set policy parameter
            Input :
                policy (str) : policy to use. sets according to anticipatory parameter on default.
            Output :
                None, sets policy parameter
        """
        # set policy according to string
        if policy and policy in ['average_policy', 'best_response', 'greedy_average_policy']:
            self.policy = policy
        else:
            self.policy = 'best_response' if np.random.uniform() <= self.anticipatory_param else 'average_policy'
        return self.policy

    def step(self, state):
        """
        Given state, produce actions to generate training data. Choose action according to set policy parameter.
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            action (int) : integer representing action id
        """

        if self.policy == 'average_policy':
            return self.ap_pick_action(state)[0]
        elif self.policy == 'best_response':
            return self.e_greedy_pick_action(state)

    def ap_pick_action(self, state):
        """

         Pick an action given a state using the average policy network
         Input:
             state (dict)
                 'obs' : actual state representation
                 'legal_actions' : possible legal actions to be taken from this state
         Output:
             action (int) : integer representing action id
         """
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            q_values = self.average_policy(state_obs)[0].cpu().detach().numpy()
            probs = remove_illegal(q_values, state['legal_actions'])
            action = np.random.choice(self.num_actions, p=probs)
            # print('sl: ', action)
            # print(q_values, action)
            return action, probs

    def predict(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            legal_actions = state['legal_actions']
            # calculate a softmax distribution over the q_values for all actions
            softmax_q_vals = self.softmax(self.dqn_eval(state_obs))[0].cpu().detach().numpy()
            predicted_action = np.argmax(softmax_q_vals)
            probs = remove_illegal(softmax_q_vals, legal_actions)
            max_action = np.argmax(probs)

        return probs, max_action

    def e_greedy_pick_action(self, state):
        """
        Pick an action given a state using epsilon greedy action selection using target network
        Makes call to e_greedy_pick_action to actually select the action
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

    def greedy_ap_pick_action(self, state):
        """
        Pick an action greedily given a state using the average policy network
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            action (int) : integer representing action id
        """
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            q_values = self.average_policy(state_obs)[0].cpu().detach().numpy()
            probs = remove_illegal(q_values, state['legal_actions'])
            action = np.argmax(probs)
            return action, probs

    def eval_step(self, state):
        """
           Pick an action given a state according to set policy. This is to be used during evaluation, so no epsilon greedy.
           Makes call to eval_pick_action or average_policy to actually select the action
           Input:
               state (dict)
                   'obs' : actual state representation
                   'legal_actions' : possible legal actions to be taken from this state
           Output:
               action (int) : integer representing action id
               probs (np.array) : softmax distribution over the actions
        """
        if self.policy == 'average_policy':
            return self.ap_pick_action(state)
        elif self.policy == 'best_response':
            return self.eval_pick_action(state)
        elif self.policy == 'greedy_average_policy':
            return self.greedy_ap_pick_action(state)

    def eval_pick_action(self, state, use_max=True):
        """
        Pick an action given a state according to max q value.
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
            use_max (bool) : should we return max action or select according to distribution
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

    def add_transition(self, transition):
        """"
        Add transition to our memory buffers and train the networks one batch.
        Input:
            transition (tuple) : tuple representation of a transition --> (state, action, reward, next state, done)
        Output:
            Nothing. Stores transition in the buffers, updates networks using memory buffers, and updates target network
            depending on what timestep we're at.
        """

        state, action, reward, next_state, done = transition
        self.rl_buffer.save(state['obs'], state['legal_actions'], action, reward, next_state['obs'], done)
        self.timestep += 1

        if self.policy == 'best_response':
            # this version saving the predicted action from dqn_eval instead of the action that was taken by the agent.
            # this prevents the reward being zeros for long time while training a lot!
            ###comment this line for next training###action = self.av_predict(state)
            self.sl_buffer.add_sa(state['obs'], action)

        if len(self.rl_buffer.memory) >= self.rl_memory_init_size and self.timestep % self.q_train_every == 0:
            rl_loss = self.train_rl()
            print(f'\r step: {self.timestep}, rl_loss on batch: {rl_loss}', end='')
            # print(f'step: {self.timestep} action value parameters updated')
        if len(self.sl_buffer.memory) >= self.sl_memory_init_size and self.timestep % self.train_every == 0:
            sl_loss = self.train_sl()
            print(f'\rAgent {self.scope}, step: {self.timestep}, sl_loss on batch: {sl_loss}', end='')
            # print(f'step: {self.timestep} average policy updated')

        if self.timestep % self.update_every == 0:
            # print('target net params updated')
            self.dqn_target.load_state_dict(self.dqn_eval.state_dict())
            self.dqn_target.eval()

    def train_rl(self):
        """
        Samples from reinforcement learning memory buffer and trains the action value network one step.
        Input:
            Nothing. Draws sample from rl buffer to train the network
        Output:
            loss (float) : loss on training batch
        """

        states, legal_actions, actions, rewards, next_states, dones = self.rl_buffer.sample()

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device)
        dones = (1 - torch.FloatTensor(dones)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        penalty = np.zeros(self.batch_size)
       

        for i in range(self.batch_size):
            if actions[i] in legal_actions[i] and actions[i] != self.num_actions - 1:
                penalty[i] += 0.2
        penalty = torch.FloatTensor(penalty).to(self.device)
        q_values = self.dqn_eval(states)
        next_q_values = self.dqn_target(next_states)
        argmax_actions = self.dqn_eval(next_states).max(1)[1].detach()
        next_q_values = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_q_values = rewards + self.discount_factor * dones * next_q_values + penalty
        expected_q_values.detach()

        self.dqn_eval.train()
        self.rl_optim.zero_grad()

        loss = self.rl_loss(q_values, expected_q_values)
        loss.backward()
        self.rl_optim.step()
        self.dqn_eval.eval()
        return loss.item()

    def train_sl(self):
        """
        Samples from supervised learning memory buffer and trains the average policy network one step.
        Input:
            Nothing. Draws sample from sl buffer to train the network
        Output:
            loss (float) : loss on training batch
        """

        samples = self.sl_buffer.sample()

        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]

        # [batch, state_shape(450)]
        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        # [batch, 1]
        actions = torch.LongTensor(actions).to(self.device)

        #### optimizing the log-prob of past actions taken as described in NFSP paper
        # [batch, action_num(309)]
        probs = self.average_policy(states)
        # [batch, 1]
        prob = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # adding a small eps to torch.log(), preventing nan in the log_prob
        eps = 1e-7
        log_prob = torch.log(prob + eps)
        # print('log_prob', log_prob)
        loss = -log_prob.mean()

        self.average_policy.train()
        self.sl_optim.zero_grad()
        loss.backward()
        self.sl_optim.step()
        self.average_policy.eval()

        return loss.item()

    def save_state_dict(self, file_path):
        """
        Save state dict for networks of NFSP agent
        Input:
            file_path (str) : string filepath to save agent at
        """
        state_dict = dict()
        state_dict['dqn_eval'] = self.dqn_eval.state_dict()
        state_dict['average_policy'] = self.average_policy.state_dict()
        state_dict['dqn_target'] = self.dqn_target.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
           Load agent parameters from filepath
           Input:
               file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.dqn_eval.load_state_dict(state_dict['dqn_eval'])
        self.average_policy.load_state_dict(state_dict['average_policy'])
        self.dqn_target.load_state_dict(state_dict['dqn_target'])



class ReservoirMemoryBuffer:
    """
    Save a series of state action pairs to use in training of average policy network
    """
    def __init__(self, max_size, batch_size, rep_prob=0.25):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = []
        self.rep_prob = rep_prob
        self.add_ = 0

    def add_sa(self, state, action):

        if len(self.memory) < self.max_size:
            self.memory.append((state, action))
        else:
            idx = np.random.randint(0, self.add_ + 1)
            if idx < self.max_size:
                self.memory[idx] = (state, action)
        self.add_ += 1
    """
    def add_sa(self, state, action):
        # reservoir sampling with exponential bias toward newer examples. Used in nfsp paper, rep_prob=0.25

        if len(self.memory) < self.max_size:
            self.memory.append((state, action))
        elif np.random.uniform() <= self.rep_prob:
            i = int(np.random.uniform() * self.max_size)
            self.memory[i] = (state, action)
    """
    def sample(self):
        return sample(self.memory, self.batch_size)

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)
