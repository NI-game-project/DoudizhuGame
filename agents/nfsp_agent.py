import numpy as np
import torch

from agents.value_based.duel_dqn_agent import DQNAgent
from agents.buffers import ReservoirMemoryBuffer
from agents.networks import AveragePolicyNet
from utils_global import remove_illegal


# action saved in sl_buffer: action taken by the agent after removing illegal or predicted by dqn_net??
# sl_loss: optimizing the log-prob of past actions taken   L = E(-log(ap_net(s,a)))


class NFSPAgent:
    """
    Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        sl_lr (float) : learning rate to use for training average policy net
        rl_lr (float) : learning rate to use for training action value net
        batch_size (int) : batch sizes to use when training networks
        sl_memory_size (int) : max number of experiences to store in supervised learning memory buffer
        epsilon_decay_steps (int) : how often should we decay epsilon value
        eta (float) : anticipatory parameter for NFSP
        gamma (float) : discount parameter
        device (torch.device) : device to put models on
    """
    def __init__(self,
                 # these are the hyperparameters in nfsp paper
                 scope,
                 num_actions,
                 state_shape,
                 sl_lr=.005,
                 rl_lr=.1,
                 batch_size=256,
                 train_every=128,
                 sl_memory_init_size=1000,
                 sl_memory_size=int(2e6),
                 q_train_every=128,
                 epsilon_decay_steps=int(1e5),
                 eta=.2,
                 gamma=.99,
                 device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.scope = scope
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.rl_lr = rl_lr
        self.sl_lr = sl_lr
        self.batch_size = batch_size
        self.train_every = train_every
        self.discount_factor = gamma
        self.sl_memory_init_size = sl_memory_init_size
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
                                               use_conv=True,).to(self.device)
        self.average_policy.eval()

        # action value and target network are Deep Q Networks
        self.rl_agent = DQNAgent(state_shape=self.state_shape,
                                 num_actions=self.num_actions,
                                 lr=self.rl_lr,
                                 batch_size=128,
                                 train_every=64,
                                 epsilons=self.epsilons,
                                 )

        # initialize optimizers
        """
        in the paper: using sgd optim, eta = 0.1.
        rl_lr = 0.1, sl_lr = 0.005,
        epsilon decay from 0.06 to 0
        """
        self.sl_optim = torch.optim.Adam(self.average_policy.parameters(), lr=self.sl_lr)

        # initialize memory buffers
        self.sl_buffer = ReservoirMemoryBuffer(sl_memory_size, batch_size)

        # current policy
        self.policy = None

        self.softmax = torch.nn.Softmax(dim=1)

        self.timestep = 0

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
            q_values = self.average_policy(state_obs)[0].cpu().detach().numpy()
            probs = remove_illegal(q_values, state['legal_actions'])
            action = np.random.choice(self.num_actions, p=probs)
            # print('sl: ', action)
            # print(q_values, action)
            return action, probs

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
        Given state, produce actions to generate training data. Choose action according to set policy parameter.
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
        if self.policy == 'average_policy':
            action, probs = self.ap_pick_action(state)
        elif self.policy == 'greedy_average_policy':
            action, probs = self.greedy_ap_pick_action(state)
        elif self.policy == 'best_response':
            action, probs = self.rl_agent.eval_step(state)
        self.actions.append(action)

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
        self.timestep += 1

        if self.policy == 'best_response':
            # this version saving the predicted action from dqn_eval instead of the action that was taken by the agent.
            self.sl_buffer.add_sa(state['obs'], action)

        if len(self.sl_buffer.memory) >= self.sl_memory_init_size and self.timestep % self.train_every == 0:
            sl_loss = self.train_sl()
            print(f'\rAgent {self.scope}, step: {self.timestep}, sl_loss on batch: {sl_loss}', end='')
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

        # [batch, state_shape(450)]
        states = torch.FloatTensor(states).to(self.device)
        # [batch, 1]
        actions = torch.LongTensor(actions).to(self.device)

        #### optimizing the log-prob of past actions taken as in NFSP paper
        # [batch, action_num(309)]
        probs = self.average_policy(states)
        # [batch, 1]
        prob = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # adding a small eps to torch.log(), avoiding nan in the log_prob
        eps = 1e-7
        log_prob = torch.log(prob + eps)

        ### look into torch.nll_loss
        loss = -log_prob.mean()

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
        state_dict['average_policy'] = self.average_policy.state_dict()
        state_dict['dqn_local'] = self.rl_agent.local_net.state_dict()
        state_dict['dqn_target'] = self.rl_agent.target_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
           Load agent parameters from filepath
           Input:
               file_path (str) : string filepath to load parameters from
        """

        state_dict = torch.load(filepath, map_location=self.device)
        self.average_policy.load_state_dict(state_dict['average_policy'])
        self.rl_agent.local_net.load_state_dict(state_dict['dqn_local'])
        self.rl_agent.target_net.load_state_dict(state_dict['dqn_target'])