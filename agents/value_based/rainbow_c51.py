import numpy as np
import torch
import torch.nn as nn

from utils_global import remove_illegal, action_mask
from agents.common.model import C51DuelDQN, C51DQN
from agents.common.buffers import NStepPERBuffer, NStepBuffer, PrioritizedBuffer, BasicBuffer
from agents.value_based.dqn_base_agent import DQNBaseAgent


class RainbowAgent(DQNBaseAgent):
    """
    An implementation of rainbow dqn agent
    with double, dueling, noisy, c51(categorical/distribution) network, multi-step prioritized replay buffer.

        Q(s,a) is the expected reward. Z is the full distribution from which Q is generated.
        Support represents the support of Z distribution (non-zero part of pdf).
        Z is represented with a fixed number of "atoms", which are pairs of values (x_i, p_i)
        composed by the discrete positions (x_i) equidistant along its support defined between
        Vmin-Vmax and the probability mass or "weight" (p_i) for that particular position.
        As an example, for a given (s,a) pair, we can represent Z(s,a) with 8 atoms as follows:
                   .        .     .
                .  |     .  |  .  |
                |  |  .  |  |  |  |  .
                |  |  |  |  |  |  |  |
           Vmin ----------------------- Vmax

       Parameters:
        Parameters:
        num_actions (int) : how many possible actions
        state_shape (list) : tensor shape of state
        num_atoms (int) : the number of buckets for the value function distribution.
        v_max (float): maximum return predicted by a value distribution.(max_reward of the game)
        v_min (float): -v_max
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
                 num_atoms=51,
                 v_min=-1.,
                 v_max=1.,
                 lr=0.00001,
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
                 loss_type=None,
                 clip=False,
                 double=True,
                 dueling=True,
                 noisy=True,
                 use_n_step=True,
                 n_step=3,
                 per=True,
                 use_conv=False,
                 device=None):

        super().__init__(state_shape, num_actions, lr, gamma, epsilon_start, epsilon_end, epsilon_decay_steps,
                         batch_size, train_every, replay_memory_size, replay_memory_init_size, soft_update,
                         soft_update_target_every, hard_update_target_every, loss_type, double, noisy, clip, use_conv,
                         device)

        self.double = double
        self.dueling = dueling
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.per = per
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # support: [num_atoms]
        # [-1., -0.96, -0.92, ..., 0.92, 0.96, 1.]
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        # delta_z: scalar(0.04)
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        # initialize online and target networks
        if dueling:
            self.online_net = C51DuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         num_atoms=self.num_atoms, use_conv=use_conv,
                                         noisy=self.noisy).to(self.device)
            self.target_net = C51DuelDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         num_atoms=self.num_atoms, use_conv=use_conv,
                                         noisy=self.noisy).to(self.device)
        else:
            self.online_net = C51DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                     num_atoms=self.num_atoms, use_conv=use_conv,
                                     noisy=self.noisy).to(self.device)
            self.target_net = C51DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                     num_atoms=self.num_atoms, use_conv=use_conv,
                                     noisy=self.noisy).to(self.device)

        self.online_net.train()
        self.target_net.train()
        # Disable calculations of gradients of the target network.
        for param in self.target_net.parameters():
            param.requires_grad = False

        # initialize optimizer(Adam) for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

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

    @staticmethod
    def KL_divergence_two_dist(dist_p, dist_q):
        kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
        return kld

    def projection_distribution(self, next_dist, rewards, dones):
        """
        Returns probability distribution for target policy given the visited transitions. Since the
        Q function is defined as a discrete distribution, the expected returns will most likely
        fall outside the support of the distribution and we won't be able to compute the KL
        divergence between the target and online policies for the visited transitions. Therefore, we
        need to project the resulting distribution into the support defined by the network output
        definition.
        """

        # [batch_size, num_atoms]
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = self.support.unsqueeze(0).expand_as(next_dist)

        # Compute projection of the application of the Bellman operator.
        # compute projected values for each particular atom zi according to: zi=r+γzi
        # For calculating Tz we use the support to calculate ALL possible expected returns,
        # i.e. the full distribution, without looking at the probabilities yet.
        # Clamp values so they fall within the support of Z values
        if self.use_n_step:
            Tz = rewards + (1 - dones) * support * (self.gamma ** self.n_step)
        else:
            Tz = rewards + (1 - dones) * support * self.gamma

        Tz = Tz.clamp(min=self.v_min, max=self.v_max)

        # Compute categorical indices for distributing the probability
        #  1. Find which values of the discrete fixed distribution are the closest lower (l) and
        #     upper value (u) to the values obtained from Tz (b). As a reminder, b is the new support
        #     of our return distribution shifted from the original network output support when we
        #     computed Tz. In other words, b is how many times deltaz from Vmin to get to
        #     Tz by definition: b = (Tz - Vmin) / Δz
        #     We've expressed Tz in terms of b (the misaligned support but still similar in the sense
        #     of exact same starting point and exact same distance between the atoms as the original
        #     support). Still misaligned but that's why do the redistribution in terms of proportionality.
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # 2. Distribute probability of Tz. Since b is most likely not having the exact value of
        # one of our predefined atoms, we split its probability mass between the closest atoms
        #   (l, u) in proportion to their OPPOSED distance to b so that the closest atom receives
        #    most of the mass in proportion to their distances.
        #                          u
        #              l    b      .
        #              ._d__.__2d__|
        #         ...  |    :      |  ...    mass_l += mass_b * 2 / 3
        #              |    :      |         mass_u += mass_b * 1 / 3
        #    Vmin ----------------------- Vmax
        #    The probability mass becomes 0 when l = b = u (b is int). Note that for this case
        #    u - b + b - l = b - b + b - b = 0
        #    To fix this, we change  l -= 1 which would result in:
        #    u - b + b - l = b - b + b - (b - 1) = 1
        #    Except in the case where b = 0, because l -=1 would make l = -1
        #    Which would mean that we are subtracting the probability mass! To handle this case we
        #    would only do l -=1 if u > 0, and for the particular case of b = u = l = 0 we would
        #    keep l = 0 but u =+ 1
        l[(u > 0) * (l == u)] -= 1  # Handles the case of u = b = l != 0
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1  # Handles the case of u = b = l = 0

        # [batch_size, num_atoms]
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long() \
            .unsqueeze(1).expand(self.batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size(), dtype=torch.float32)
        # Distribute probabilities to the closest lower atom in inverse proportion to the
        # distance to the atom. For efficiency, we are adding the values to the flattened view of
        # the array not flattening the array itself.
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

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
            # Distribution of the probabilities of θ(s,a) on the support
            dist = self.online_net(state_obs).cpu().detach()
            dist = dist * self.support.expand_as(dist)

            # get q_values by summing up over the distribution of each action,
            # then calculate a softmax distribution over q_values
            q_values = dist.sum(2)[0]
            softmax_q_vals = self.softmax(q_values).numpy()

            predicted_action = np.argmax(q_values.numpy())
            probs = remove_illegal(softmax_q_vals, legal_actions)
            max_action = np.argmax(probs)

            self.q_values = q_values

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

        # Calculate value distributions of current (states, actions).
        dists = self.online_net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dists = dists.gather(1, actions).squeeze(1)
        # trick for avoiding nans
        dists.detach().data.clamp_(0.01, 0.99)

        with torch.no_grad():
            # Calculate value distributions of next states.
            next_dist_target = self.target_net(next_states)

            # next_dist: [batch_size, num_actions, num_atoms]
            if self.double:
                # Reset noise of online network to decorrelate between action selection and value distribution calculation.
                if self.noisy:
                    self.reset_noise()
                # use online network to select next argmax action
                next_dist_online = self.online_net(next_states)
            else:
                # use target network to select next argmax action
                next_dist_online = next_dist_target

            # get q_values by summing up over the last dim of distribution
            # next_q_values: [batch_size, num_actions]
            next_q_values = (next_dist_online * self.support.expand_as(next_dist_online)).sum(2)

            # Do action mask for q_values of next_state if not done (i.e., set q_values of illegal actions to -inf)
            for i in range(self.batch_size):
                next_q_values[i] = action_mask(self.num_actions, next_q_values[i], next_legal_actions[i])

            # Select greedy actions a∗ in next state using the target(online if double) network., a∗=argmaxa′Qθ(s′,a′)
            next_argmax_action = next_q_values.max(1)[1]
            # next_argmax_action: [batch_size, 1, num_atoms]
            next_argmax_action = next_argmax_action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
            # Use greedy actions to select target value distributions.
            # next_dist: [batch_size, num_atoms]
            next_dist = next_dist_target.gather(1, next_argmax_action).squeeze(1)

            # project next value distribution onto the support
            proj_dists = self.projection_distribution(next_dist, rewards, dones)

        # Cross-entropy loss (minimises KL-distance between online and target probs): DKL(proj_dists || dists)
        # dists: policy distribution for online network
        # proj_dists: aligned policy distribution for target network
        error = -(proj_dists * dists.log()).sum(1)

        if self.per:
            # update per priorities
            self.memory_buffer.update_priorities(indices, abs(error).detach().numpy())
            # calculate importance-weighted (Prioritized Experience Replay) batch loss
            loss = (error * is_weights).mean()
        else:
            loss = error.mean()

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
        self.train_time += 1

        self.expected_q_values = (proj_dists * self.support).sum(1)
        self.current_q_values = (dists * self.support).sum(1)

        return loss.item()
