import torch
import torch.nn as nn

from utils_global import action_mask
from agents.common.model import DQN, DuelingDQN, DeepConvNet
from agents.common.buffers import PrioritizedBuffer, BasicBuffer
from agents.value_based.dqn_base_agent import DQNBaseAgent

"""
An implementation of PER/Noisy/Dueling/Double dqn Agent
"""


class PERDQNAgent(DQNBaseAgent):
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
        dueling (bool) : if use dueling structure for network
        use_conv (bool) : if use convolutional layers for network
        device (torch.device) : device to put models on
    """

    def __init__(self,
                 state_shape,
                 num_actions,
                 lr=0.00001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=40000,
                 batch_size=32,
                 train_every=1,
                 replay_memory_size=int(2e4),
                 replay_memory_init_size=1000,
                 soft_update=False,
                 soft_update_target_every=10,
                 hard_update_target_every=1000,
                 loss_type='mse',
                 double=True,
                 dueling=False,
                 noisy=False,
                 clip=False,
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
                         soft_update=soft_update,
                         soft_update_target_every=soft_update_target_every,
                         hard_update_target_every=hard_update_target_every,
                         loss_type=loss_type,
                         double=double,
                         noisy=noisy,
                         clip=clip,
                         use_conv=use_conv,
                         device=device)

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

        # initialize optimizer for online network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.online_net.train()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # initialize memory buffer
        self.memory_buffer = PrioritizedBuffer(replay_memory_size, batch_size)

    def train(self):
        """
        Samples from memory buffer to get a batch of samples with its weights probability distribution
        and trains the network one step.
        Input:
            Nothing. Draws sample from memory buffer to train the network
        Output:
            loss (float) : loss on training batch

        """

        states, legal_actions, actions, rewards, next_states, next_legal_actions, dones, indices, is_weights \
            = self.memory_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # importance sampling weights
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        self.online_net.train()
        self.target_net.train()

        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Predict its Q-value Qθ′(s′,a∗) using the target network.
            next_q_values_target = self.target_net(next_states)

            # Sample the noise of online network to decorrelate between action selection and quantile calculation.
            if self.double:
                if self.noisy:
                    self.reset_noise()
            # Select the greedy action in the next state a∗=argmaxa′Qθ(s′,a′) using the local network.
                next_q_values_online = self.online_net(next_states)
            else:
                next_q_values_online = next_q_values_target

            # Do action_mask for q_values of next_state if not done
            for i in range(self.batch_size):
                next_q_values_online[i] = action_mask(self.num_actions, next_q_values_online[i], next_legal_actions[i])

            next_argmax_actions = next_q_values_online.max(1)[1]

            next_q_values = next_q_values_target.gather(1, next_argmax_actions.unsqueeze(1)).squeeze(1)

        # Compute the target value y=r+γQθ′(s′,a∗)
        # value = reward + gamma * target_network.predict(next_state)[argmax(local_network.predict(next_state))]
        expected_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        expected_q_values.detach()

        # calculate loss and update priorities
        # an error of a sample is the distance between the Q(s, a) and its target T(S): error=|Q(s,a)−T(S)|
        # store this error in the agent’s memory along with sample index and update it each learning step.
        td_error = torch.abs(expected_q_values - q_values)
        self.memory_buffer.update_priorities(indices, td_error.detach().numpy())

        # calculate the MSEloss
        error = torch.pow(td_error, 2)
        loss = (error * is_weights).mean()

        self.optimizer.zero_grad()
        # Backpropagate importance-weighted (Prioritized Experience Replay) minibatch loss
        loss.backward()

        if self.clip:
            # Clip gradients (normalising by max value of gradient L1 norm)
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.clip_norm)
            # nn.utils.clip_grad_value_(self.online_net.parameters(), clip_value=0.5)
        self.optimizer.step()

        self.loss = loss.item()

        # soft/hard update the parameters of the target network and increase the training time
        self.update_target_net(self.soft_update)
        self.train_time += 1

        self.current_q_values = q_values
        self.expected_q_values = expected_q_values

        return loss.item()
