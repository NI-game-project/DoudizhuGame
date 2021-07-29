import numpy as np
import torch
import torch.nn as nn

from agents.value_based.utils import disable_gradients, gaussian_fn
from utils_global import action_mask
from agents.common.model import MoGDQN, MoGDuelDQN
from agents.common.buffers import NStepPERBuffer, NStepBuffer, PrioritizedBuffer, BasicBuffer
from agents.value_based.dqn_base_agent import DQNBaseAgent


class RainbowAgent(DQNBaseAgent):
    """
    An implementation of rainbow dqn agent
    with double, dueling, noisy, mog_dqn(Mixture of Gaussians) network, multi-step prioritized replay buffer.

       Parameters:
        Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        num_gaussians (int) : number of Gaussian distributions
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
        use_n_step (bool) : if n_step buffer for storing experience is used
        n_step (int) : how many steps of information to store in buffer
        clip (bool) : if gradient is clipped(norm / value)
        double (bool) : if use double dqn(i.e., use online network to select next_argmax_action)
        use_conv (bool) : if use convolutional layers for network
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 num_gaussians=5,
                 lr=0.00001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=int(1e5),
                 epsilon_eval=0.001,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(1e5),
                 replay_memory_init_size=1000,
                 hard_update_target_every=1000,
                 hidden_size=512,
                 double=True,
                 dueling=True,
                 noisy=True,
                 use_n_step=True,
                 n_step=3,
                 per=True,
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
                         epsilon_eval=epsilon_eval,
                         batch_size=batch_size,
                         train_every=train_every,
                         replay_memory_size=replay_memory_size,
                         replay_memory_init_size=replay_memory_init_size,
                         hard_update_target_every=hard_update_target_every,
                         double=double,
                         noisy=noisy,
                         clip=clip,
                         device=device)

        self.double = double
        self.dueling = dueling
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.per = per
        self.num_gaussians = num_gaussians

        # initialize online and target networks
        if dueling:
            self.online_net = MoGDuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         num_gaussians=self.num_gaussians, hidden_size=hidden_size,
                                         use_conv=use_conv, noisy=self.noisy)
            self.target_net = MoGDuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         num_gaussians=self.num_gaussians, hidden_size=hidden_size,
                                         use_conv=use_conv, noisy=self.noisy).to(self.device)
        else:
            self.online_net = MoGDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                     num_gaussians=self.num_gaussians, hidden_size=hidden_size,
                                     use_conv=use_conv, noisy=self.noisy)
            self.target_net = MoGDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                     num_gaussians=self.num_gaussians, hidden_size=hidden_size,
                                     use_conv=use_conv, noisy=self.noisy).to(self.device)

        self.online_net.train()
        self.target_net.train()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        # initialize optimizer(Adam) for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, )  # eps=0.00015

        # initialize memory buffer
        if self.n_step:
            if self.per:
                self.memory_buffer = NStepPERBuffer(replay_memory_size, batch_size, self.n_step, self.gamma)
            else:
                self.memory_buffer = NStepBuffer(replay_memory_size, batch_size, self.n_step, self.gamma)
        elif self.per:
            self.memory_buffer = PrioritizedBuffer(replay_memory_size, batch_size)
        else:
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
            # weight, mean, variance of the distribution of θ(s,a)
            pi, mu, sigma = self.online_net(state_obs)
            pi.detach()
            mu.detach()
            sigma.detach()

            # get q_values: the expected value of the value dist i.e., the mixture weighted sum of Gaussian means
            q_values = (pi * mu).sum(-1)[0]

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
        if self.per:
            states, legal_actions, actions, rewards, next_states, next_legal_actions, dones, indices, is_weights \
                = self.memory_buffer.sample()
            # importance sampling weights
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            states, legal_actions, actions, rewards, next_states, next_legal_actions, dones = self.memory_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        self.online_net.train()
        self.target_net.train()

        # pi, mu, sigma: [batch_size, num_actions, num_gaussians]
        pi, mu, sigma = self.online_net(states)
        # actions: [batch_size, 1, num_gaussians]
        actions = actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        # [batch_size, num_gaussians]
        pi = pi.gather(1, actions).squeeze()
        mu = mu.gather(1, actions).squeeze()
        sigma = sigma.gather(1, actions).squeeze()

        with torch.no_grad():
            next_pi_target, next_mu_target, next_sigma_target = self.target_net(next_states)

            if self.double:
                # reset the noise of online network to decorrelate between action selection and quantile calculation.
                if self.noisy == 'noisy':
                    self.reset_noise()
                # use online network to select next argmax action
                next_pi_online, next_mu_online, next_sigma_online = self.online_net(next_states)
            else:
                # use target network to select next argmax action
                next_pi_online, next_mu_online, next_sigma_online = next_pi_target, next_mu_target, next_sigma_target

            # get next q values by the calculating the mixture weighted sum of Gaussian means
            next_q_values = (next_pi_online * next_mu_online).sum(-1)

            # Do action mask for q_values of next_state if not done (i.e., set q_values of illegal actions to -inf)
            for i in range(self.batch_size):
                next_q_values[i] = action_mask(self.num_actions, next_q_values[i], next_legal_actions[i])

            # Select greedy actions a∗ in next state using the target(online if double) network., a∗=argmaxa′Qθ(s′,a′)
            next_argmax_actions = next_q_values.max(1)[1].detach()
            # next_argmax_actions: [batch_size, 1, num_gaussians]
            next_argmax_actions = next_argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, self.num_gaussians])

            # Use greedy actions to select target gaussian parameters.
            next_pi = next_pi_target.gather(1, next_argmax_actions).squeeze()
            next_mu = next_mu_target.gather(1, next_argmax_actions).squeeze()
            next_sigma = next_sigma_target.gather(1, next_argmax_actions).squeeze()

            # mean is shifted as mu -> gamma*mu + r
            next_mu = self.gamma * next_mu * (1. - dones.unsqueeze(-1)) + rewards.unsqueeze(-1)
            # variance is shifted as sigma**2 -> (gamma*sigma)**2
            next_sigma = self.gamma * self.gamma * next_sigma
            next_pi = next_pi.detach()
            next_mu = next_mu.detach()
            next_sigma = next_sigma.detach()

        self.expected_q_values = (next_pi * next_mu).sum(-1)
        self.current_q_values = (pi * mu).sum(-1)

        # compute the JTD loss to measure the distance between two mixtures of Gaussians.
        pi_i = pi.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        pi_j = pi.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        mu_i = mu.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        mu_j = mu.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        sigma_i = sigma.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        sigma_j = sigma.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        next_pi_i = next_pi.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        next_pi_j = next_pi.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        next_mu_i = next_mu.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        next_mu_j = next_mu.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        next_sigma_i = next_sigma.unsqueeze(-1).repeat([1, 1, self.num_gaussians])
        next_sigma_j = next_sigma.unsqueeze(-2).repeat([1, self.num_gaussians, 1])

        # equation(3) from the paper
        jtd_loss = (pi_i * pi_j * gaussian_fn(mu_j, sigma_i + sigma_j, mu_i)).sum(-1).sum(-1) + \
                   (next_pi_i * next_pi_j * gaussian_fn(next_mu_j, next_sigma_i + next_sigma_j, next_mu_i)).sum(-1).sum(
                       -1) - \
                   2 * (pi_i * next_pi_j * gaussian_fn(next_mu_j, sigma_i + next_sigma_j, mu_i)).sum(-1).sum(-1)

        if self.per:
            # update per_double_dqn priorities
            priorities = torch.abs(jtd_loss).detach().numpy()
            self.memory_buffer.update_priorities(indices, priorities)
            # calculate importance-weighted (Prioritized Experience Replay) batch loss
            loss = (jtd_loss * is_weights).mean()
        else:
            loss = jtd_loss.mean()

        self.optimizer.zero_grad()

        # Backpropagate importance-weighted (Prioritized Experience Replay) batch loss
        loss.backward()

        # Clip gradients (normalising by max value of gradient L2 norm)
        if self.clip:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.clip_norm)
            # nn.utils.clip_grad_value_(self.online_net.parameters(), clip_value=0.5)

        self.optimizer.step()

        self.loss = loss.item()

        # soft/hard update the parameters of the target network
        self.update_target_net(self.soft_update)
        self.train_step += 1

        return loss.item()
