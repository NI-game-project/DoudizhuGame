import torch
import torch.nn as nn

from agents.value_based.utils import disable_gradients
from utils_global import action_mask
from agents.common.model import DQN, DuelingDQN, DeepConvNet
from agents.common.buffers import BasicBuffer, NStepBuffer
from agents.value_based.dqn_base_agent import DQNBaseAgent

"""
An implementation of NStep/Noisy/Dueling/Double dqn Agent
"""


class NStepDQNAgent(DQNBaseAgent):
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
        hard_update_target_every (int): how often to hard update the target network(copy the param of online network)
        loss_type (str) : which loss to use, ('mse' / 'huber')
        noisy (bool) : True if use NoisyLinear for network False Linear
        use_n_step (bool) : if n_step buffer for storing experience is used
        n_step (int) : how many steps of information to store in buffer
        clip (bool) : if gradient is clipped(norm / value)
        dueling (bool) : if use dueling structure for network
        use_conv (bool) : if use convolutional layers for network
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 lr=0.0001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(1e5),
                 replay_memory_init_size=1000,
                 hard_update_target_every=1000,
                 loss_type='huber',
                 double=True,
                 dueling=False,
                 noisy=False,
                 use_n_step=False,
                 n_step=3,
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
                         loss_type=loss_type,
                         double=double,
                         noisy=noisy,
                         clip=clip,
                         use_conv=use_conv,
                         device=device)

        self.use_n_step = use_n_step
        self.n_step = n_step

        # initialize online and target networks
        if dueling:
            self.online_net = DuelingDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         use_conv=self.use_conv, noisy=self.noisy).to(self.device)
            self.target_net = DuelingDQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                         use_conv=self.use_conv, noisy=self.noisy).to(self.device)
        else:
            self.online_net = DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                  use_conv=self.use_conv, noisy=self.noisy).to(self.device)
            self.target_net = DQN(state_shape=self.state_shape, num_actions=self.num_actions,
                                  use_conv=self.use_conv, noisy=self.noisy).to(self.device)

        self.online_net.train()
        self.target_net.train()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        # initialize optimizer for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # initialize memory buffer
        if use_n_step:
            self.memory_buffer = NStepBuffer(replay_memory_size, batch_size, self.n_step, self.gamma)
        else:
            self.memory_buffer = BasicBuffer(replay_memory_size, batch_size)

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

            if self.double:
                if self.noisy:
                    # Reset noise of online network to decorrelate between action selection and q values calculation.
                    self.reset_noise()
                # use online network to select next argmax action
                next_q_values_online = self.online_net(next_states)
            else:
                # use target network to select next argmax action
                next_q_values_online = next_q_values_target

            # Do action mask for q_values of next_state if not done (i.e., set q_values of illegal actions to -inf)
            for i in range(self.batch_size):
                next_q_values_online[i] = action_mask(self.num_actions, next_q_values_online[i], next_legal_actions[i])

            # Select greedy actions a∗ in next state using the target(online if double) network., a∗=argmaxa′Qθ(s′,a′)
            next_argmax_actions = next_q_values_online.max(1)[1]
            # Predict its Q-value Qθ′(s′,a∗) using the target network.
            next_q_values = next_q_values_target.gather(1, next_argmax_actions.unsqueeze(1)).squeeze(1)

        # Compute the expected q value y=r+γQθ′(s′,a∗)
        # value = reward + gamma * target_network.predict(next_state)[argmax(online_network.predict(next_state))]
        if self.use_n_step:
            expected_q_values = rewards + (self.gamma ** self.n_step) * (1 - dones) * next_q_values
        else:
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
