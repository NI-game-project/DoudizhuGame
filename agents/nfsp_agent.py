import numpy as np
import torch

from agents.value_based.dqn_base_agent import DQNBaseAgent
from agents.value_based.rainbow_c51 import RainbowAgent
from agents.common.buffers import ReservoirMemoryBuffer
from agents.common.model import AveragePolicyNet
from utils_global import remove_illegal


class NFSPAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        sl_lr (float) : learning rate to use for training average policy net
        rl_lr (float) : learning rate to use for training action value net
        batch_size (int) : batch sizes to use when training networks
        sl_memory_size (int) : max number of experiences to store in supervised learning memory buffer
        eta (float) : anticipatory parameter for NFSP
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 num_actions,
                 state_shape,
                 sl_lr=.001,
                 rl_lr=.0005,
                 batch_size=128,
                 train_every=4,
                 sl_memory_init_size=1000,
                 sl_memory_size=int(1e6),
                 eta=.2,
                 device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.sl_lr = sl_lr
        self.batch_size = batch_size
        self.train_every = train_every
        self.sl_memory_init_size = sl_memory_init_size
        self.anticipatory_param = eta
        self.use_raw = False

        # average policy can be modeled as a Deep Q Network and we take softmax after final layer
        self.average_policy = AveragePolicyNet(state_shape=state_shape,
                                               num_actions=num_actions,
                                               use_conv=False).to(self.device)
        self.average_policy.eval()

        # initialize inner RL agent
        self.rl_agent = DQNBaseAgent(state_shape=self.state_shape,
                                     num_actions=self.num_actions,
                                     lr=rl_lr,
                                     batch_size=64,
                                     train_every=4,
                                     epsilon_start=0.06,
                                     epsilon_end=0,
                                     epsilon_decay_steps=int(1e5),
                                     )

        # initialize optimizers
        self.sl_optim = torch.optim.Adam(self.average_policy.parameters(), lr=self.sl_lr)

        # initialize memory buffers
        self.sl_buffer = ReservoirMemoryBuffer(sl_memory_size, batch_size)

        # current policy
        self.policy = None

        self.total_step = 0
        self.train_step = 0

        # for plotting
        self.loss = 0
        self.actions = []
        self.predictions = []

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
            state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
            legal_actions = state['legal_actions']
            action_probs = self.average_policy(state_obs)[0].cpu().detach().numpy()
            prediction = np.argmax(action_probs)
            probs = remove_illegal(action_probs, legal_actions)
            action = np.random.choice(self.num_actions, p=probs)

            return action, probs, prediction

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
            state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
            q_values = self.average_policy(state_obs)[0].cpu().detach().numpy()
            probs = remove_illegal(q_values, state['legal_actions'])
            action = np.argmax(probs)

            return action, probs

    def step(self, state):
        """
        Given state, produce actions to generate training data. Choose action according to policy parameter.
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            action (int) : integer representing action id
        """

        if self.policy == 'average_policy':
            action = self.ap_pick_action(state)[0]
        elif self.policy == 'best_response':
            action = self.rl_agent.step(state)

        return action

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

        self.average_policy.eval()

        if self.policy == 'average_policy':
            action, probs, predicted_action = self.ap_pick_action(state)
            self.predictions.append(predicted_action)
        elif self.policy == 'greedy_average_policy':
            action, probs = self.greedy_ap_pick_action(state)
        elif self.policy == 'best_response':
            action, probs = self.rl_agent.eval_step(state)
        self.actions.append(action)

        self.average_policy.train()

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
        self.rl_agent.add_transition(transition)
        self.total_step += 1

        if self.policy == 'best_response':
            # Store (state, action) to Reservoir Buffer for Supervised Learning
            self.sl_buffer.add_sa(state['obs'], action)

        if len(self.sl_buffer.memory) >= self.sl_memory_init_size and self.total_step % self.train_every == 0:
            sl_loss = self.train_sl()
            print(f'\rstep: {self.total_step}, sl_loss on batch: {sl_loss}', end='')
            # print(f'step: {self.timestep} average policy updated')

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

        self.average_policy.train()
        self.sl_optim.zero_grad()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        # [batch, action_num]
        probs = self.average_policy(states)
        # [batch, 1]
        prob = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # adding a small eps to torch.log(), avoiding nan in the log_prob
        eps = 1e-7
        # sl_loss: optimizing the log-prob of past actions taken: L = E(-log(ap_net(s,a)))
        log_prob = torch.log(prob + eps)

        loss = -log_prob.mean()

        self.loss = loss.item()
        loss.backward()
        self.sl_optim.step()

        self.train_step += 1

        return loss.item()

    def save_state_dict(self, file_path):
        """
        Save state dict for networks of NFSP agent
        Input:
            file_path (str) : string filepath to save agent at
        """
        state_dict = dict()
        state_dict['average_policy'] = self.average_policy.state_dict()
        state_dict['rl_online'] = self.rl_agent.online_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
           Load agent parameters from filepath
           Input:
               file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.average_policy.load_state_dict(state_dict['average_policy'])
        self.rl_agent.online_net.load_state_dict(state_dict['rl_online'])
