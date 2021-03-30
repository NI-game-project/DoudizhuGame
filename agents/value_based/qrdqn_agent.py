import numpy as np
import torch
import torch.nn as nn

from agents.value_based.utils import calculate_huber_loss, disable_gradients, calculate_quantile_loss_penalties
from utils_global import action_mask
from agents.common.model import QRDQN, QRDuelDQN
from agents.common.buffers import BasicBuffer
from agents.value_based.dqn_base_agent import DQNBaseAgent
"""
An implementation of Quantile Regression DQN Agent.
Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
"""


class QRDQNAgent(DQNBaseAgent):
    """

       Parameters:
        Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        n_quantile (int) : number of quantiles
        kappa (float) : smoothing parameter for the Huber loss
        lr (float) : learning rate to use for training online_net
        gamma (float) : discount parameter
        epsilon_start (float) : start value of epsilon
        epsilon_end (float) : stop value of epsilon
        epsilon_decay_steps (int) : how often should we decay epsilon value
        batch_size (int) : batch sizes to use when training networks
        train_every (int) : how often to update the online work
        replay_memory_init_size (int) : minimum number of experiences to start training
        replay_memory_size (int) : max number of experiences to store in memory buffer
        hard_update_target_every (int): how often to hard update the target network(copy the param of online network)
        noisy (bool) : True if use NoisyLinear for network False Linear
        n_step (int) : how many steps of information to store in buffer
        clip (bool) : if gradient is clipped(norm / value)
        double (bool) : if use double dqn(i.e., use online network to select next_argmax_action)
        dueling (bool) : if use dueling dqn network
        use_conv (bool) : if use convolutional layers for network
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 n_quantile=200,
                 kappa=1.,
                 lr=0.00001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(2e5),
                 replay_memory_init_size=1000,
                 hard_update_target_every=1000,
                 double=True,
                 dueling=False,
                 noisy=False,
                 clip=True,
                 use_conv=False,
                 device=None):

        super().__init__(state_shape=state_shape,
                         num_actions=num_actions,
                         lr=lr,
                         gamma=gamma,
                         epsilon_start=epsilon_start,
                         epsilon_end=epsilon_end,
                         epsilon_decay_steps=epsilon_decay_steps,
                         batch_size=batch_size,
                         train_every=train_every,
                         replay_memory_size=replay_memory_size,
                         replay_memory_init_size=replay_memory_init_size,
                         hard_update_target_every=hard_update_target_every,
                         double=double,
                         noisy=noisy,
                         clip=clip,
                         use_conv=use_conv,
                         device=device)

        self.n_quantile = n_quantile
        self.kappa = kappa
        # Quantile midpoints.
        # tau(i) = i/N
        # tau_hat(i) = (tau(i-1) + tau(i))/2
        self.tau_hat = (torch.arange(self.n_quantile, dtype=torch.float32) + 0.5) / self.n_quantile

        # initialize q and target networks
        if dueling:
            self.online_net = QRDuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                        n_quantile=self.n_quantile, use_conv=use_conv,
                                        noisy=self.noisy).to(self.device)
            self.target_net = QRDuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                        n_quantile=self.n_quantile, use_conv=use_conv,
                                        noisy=self.noisy).to(self.device)
        else:
            self.online_net = QRDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                    n_quantile=self.n_quantile, use_conv=use_conv,
                                    noisy=self.noisy).to(self.device)
            self.target_net = QRDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                    n_quantile=self.n_quantile, use_conv=use_conv,
                                    noisy=self.noisy).to(self.device)

        self.online_net.train()
        self.target_net.train()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        # initialize optimizer(Adam) for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=1e-2/batch_size)

        # initialize memory buffer
        self.memory_buffer = BasicBuffer(replay_memory_size, batch_size)

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
        state_obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
        legal_actions = state['legal_actions']

        with torch.no_grad():
            # Distribution of the probabilities of θ(s,a)
            dist = self.online_net(state_obs).cpu().detach()

            # get q_values by averaging over the distribution of each action
            q_values = dist.mean(2)[0]

            # Do action mask for q_values. i.e., set q_values of illegal actions to -inf
            probs = action_mask(self.num_actions, q_values, legal_actions)
            max_action = np.argmax(probs.numpy())
            predicted_action = np.argmax(q_values.numpy())

            # self.q_values = q_values.numpy()

        return probs, max_action, predicted_action

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

        # Calculate quantile values of current states and actions at taus.
        dists = self.online_net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.n_quantile)
        # [batch_size, n_quantile]
        dists = dists.gather(1, actions).squeeze(1)
        # [batch_size, n_quantile, 1]
        dists = dists.unsqueeze(2)

        # Calculate quantile values of next states and actions at tau_hats.
        with torch.no_grad():
            next_dist_target = self.target_net(next_states)

            # next_dist: [batch_size, num_actions, n_quantile]
            if self.double:
                # reset the noise of online network to decorrelate between action selection and quantile calculation.
                if self.noisy == 'noisy':
                    self.reset_noise()
                # use online network to select next argmax action
                next_dist_online = self.online_net(next_states)
            else:
                # use target network to select next argmax action
                next_dist_online = next_dist_target

            # get next q values by averaging over last dim of the distribution
            next_q_values = next_dist_online.mean(2)

            # Do action mask for q_values of next_state if not done (i.e., set q_values of illegal actions to -inf)
            for i in range(self.batch_size):
                next_q_values[i] = action_mask(self.num_actions, next_q_values[i], next_legal_actions[i])

            # Select greedy actions a∗ in next state using the target(online if double) network., a∗=argmaxa′Qθ(s′,a′)
            next_argmax_actions = next_q_values.max(1)[1].detach()
            # next_argmax_action: [batch_size, 1,  n_quantile]
            next_argmax_actions = next_argmax_actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.n_quantile)
            # Use greedy actions to select target value distributions.
            # next_dist: [batch_size, n_quantile]
            next_dists = next_dist_target .gather(1, next_argmax_actions).squeeze(1)

            # Calculate target quantile values.
            # [batch_size, n_quantile]
            target_dists = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_dists
            # [batch_size, 1, n_quantile]
            target_dists = target_dists.unsqueeze(1)

        # compute the loss between all pairs(θi, θj) of quantile of predicted and target distribution
        # [batch_size, n_quantile, n_quantile]
        td_error = target_dists.detach() - dists

        # Compute quantile penalties
        quantile_penalties = calculate_quantile_loss_penalties(u=td_error, tau=self.tau_hat)
        # Compute huber loss element-wisely.
        # [batch_size, n_quantile, n_quantile]
        huber_loss = calculate_huber_loss(u=td_error, kappa=self.kappa)
        # Compute quantile huber loss element-wisely
        # quantile regression loss penalizes overestimation errors with weight tau and underestimation errors with
        # weight 1-tau
        quantile_huber_loss = (quantile_penalties * huber_loss) / self.kappa
        # Average over target value dimension, sum over tau dimension.
        # quantile_loss = quantile_huber_loss.sum(1).mean(1)
        quantile_loss = torch.sum(torch.mean(quantile_huber_loss, 2), 1)

        # Average loss over the batch
        loss = quantile_loss.mean()
        self.optimizer.zero_grad()

        loss.backward()

        # Clip gradients (normalising by max value of gradient L2 norm)
        if self.clip:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.clip_norm)
            # nn.utils.clip_grad_value_(self.online_net.parameters(), clip_value=0.5)

        self.optimizer.step()

        self.loss = loss.item()

        # soft/hard update the parameters of the target network
        self.update_target_net(self.soft_update)
        self.train_time += 1

        self.expected_q_values = target_dists.mean(2)[0]
        self.current_q_values = dists.mean(1)[0]

        return loss.item()


