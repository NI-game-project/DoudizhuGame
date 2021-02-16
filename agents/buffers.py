from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemoryBuffer(object):
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

    def save(self, state, action, reward, next_state, done):
        """
            Save transition into memory

            Args:
                state (numpy.array): the current state
                action (int): the performed action ID
                reward (float): the reward received
                next_state (numpy.array): the next state after performing the action
                done (boolean): whether the episode is finished
        """

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        """
            Sample a minibatch from the replay memory

            Returns:
                state_batch (list): a batch of states
                action_batch (list): a batch of actions
                reward_batch (list): a batch of rewards
                next_state_batch (list): a batch of states
                done_batch (list): a batch of dones
        """
        # TODO: Add legal_action_batch to the memory
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)


Transition_ = namedtuple('Transition', ['state', 'legal_actions', 'action', 'reward', 'next_state', 'done'])


class ReplayMemoryBuffer_(object):
    """
    Memory for saving transitions
    Add legal_actions to buffer used to calculate bonus(predicting legal actions) for network
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
                legal_actions (list): a list of legal actions - for calculating bonus for the network
                action (int): the performed action ID
                reward (float): the reward received
                next_state (numpy.array): the next state after performing the action
                done (boolean): whether the episode is finished
        """

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition_(state, legal_actions, action, reward, next_state, done)
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

    def __iter__(self):
        return iter(self.memory)


class SequentialMemory(object):
    """
        sequential memory implementation for recurrent q network
        save a series of transitions to use as training examples for the recurrent network
    """
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = []

    def add_seq_transition(self, seq):
        if len(self.memory) == self.max_size:
            self.memory.pop(0)
        self.memory.append(seq)

    def sample(self):
        return random.sample(self.memory, self.batch_size)


class ReservoirMemoryBuffer(object):
    """
    Save a series of state action pairs to use in training of average policy network
    For supervised learning data (state, action) pairs in NFSPAgent
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
        # Reservoir sampling implementation with exponential bias toward newer examples(in NFSP paper). rep_prob=0.25
        # this might lead to noisy performance stated in the paper

        if len(self.memory) < self.max_size:
            self.memory.append((state, action))
        elif np.random.uniform() <= self.rep_prob:
            i = int(np.random.uniform() * self.max_size)
            self.memory[i] = (state, action)
    """
    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)


class NStepReplayBuffer(object):
    def __init__(self, capacity, batch_size, n_step, gamma):
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for _, _, rewards, next_states, done in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward * (1 - done) + rewards
            next_state, done = (next_states, done) if done else (next_state, done)
        return reward, next_state, done

    def save(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.n_step_buffer.append([state, action, reward, next_state, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][: 2]
        self.memory.append([state, action, reward, next_state, done])

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*samples)
        return np.concatenate(state, 0), action, reward, np.concatenate(next_state, 0), done

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment_step=1000):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1 - beta) / beta_increment_step
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def save(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([state, action, reward, next_state, done])
        else:
            self.memory[self.pos] = [state, action, reward, next_state, done]
        self.priorities[self.pos] = max_prior
        self.pos += 1
        self.pos = self.pos % self.capacity

    def sample(self):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (-self.beta)

        if self.beta < 1:
            self.beta += self.beta_increment
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)
        return np.concatenate(state, 0), action, reward, np.concatenate(next_state, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
