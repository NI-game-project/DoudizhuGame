from collections import namedtuple, deque
import random
import numpy as np

Transition_ = namedtuple('Transition', ['state', 'legal_actions', 'action', 'reward',
                                        'next_state', 'next_legal_actions', 'done'])


class BasicBuffer(object):
    """
    Memory for saving transitions
    Add legal_actions to buffer used to calculate penalty(if predicting illegal actions) for network
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

    def save(self, state, legal_actions, action, reward, next_state, next_legal_actions, done):
        """
            Save transition into memory

            Args:
                state (numpy.array): the current state
                legal_actions (list): a list of legal actions for state
                action (int): the performed action ID
                reward (float): the reward received
                next_state (numpy.array): the next state after performing the action
                next_legal_actions (list): a list of legal actions for next_state
                done (boolean): whether the episode is finished
        """

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition_(state, legal_actions, action, reward, next_state, next_legal_actions, done)
        self.memory.append(transition)

    def sample(self):
        """
            Sample a minibatch from the replay memory

            Returns:
                state_batch (list): a batch of states
                legal_actions(list): a batch of legal_actions for state
                action_batch (list): a batch of actions
                reward_batch (list): a batch of rewards
                next_state_batch (list): a batch of states
                next_legal_actions(list): a batch of legal_actions for next_state
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


class NStepBuffer(object):
    """
    A trade-off between MC and TD,
    uses the n next immediate rewards and approximates the rest with the value of the state visited n steps later.
    """

    def __init__(self, capacity, batch_size, n_step, gamma):
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        # reward = sum([self.n_step_buffer[i][3] * (self.gamma ** i) for i in range(self.n_step)])

        reward, next_state, next_legal_actions, done = self.n_step_buffer[-1][-4:]
        for _, _, _, reward_, next_state_, _, done_ in reversed(list(self.n_step_buffer)[: -1]):
            reward = reward_ + self.gamma * reward * (1 - done_)
            next_state, done = (next_state_, done_) if done_ else (next_state, done)

        return reward, next_state, next_legal_actions, done

    def save(self, state, legal_actions, action, reward, next_state, next_legal_actions, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.n_step_buffer.append([state, legal_actions, action, reward, next_state, next_legal_actions, done])

        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, next_legal_actions, done = self._get_n_step_info()
        state, legal_actions, action = self.n_step_buffer[0][: 3]
        self.memory.append([state, legal_actions, action, reward, next_state, next_legal_actions, done])

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        state, legal_actions, action, reward, next_state, next_legal_actions, done = zip(*samples)
        return np.concatenate(state, 0), legal_actions, action, reward, \
               np.concatenate(next_state, 0), next_legal_actions, done

    def __len__(self):
        return len(self.memory)


class SumTree:
    """
    store samples in unsorted sum tree - a binary tree data structure where the parent’s value is the sum of its children.
    The samples themselves are stored in the leaf nodes.
    """

    def __init__(self, capacity):
        # number of leaf nodes that will store experiences
        self.capacity = capacity
        # 2*size of leaf(capacity) -1(root node), each node has max 2 children
        self.tree = np.zeros(2 * capacity - 1)
        # store the actual experiences
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.data_pointer = 0

    def add(self, priority, data):
        """
        store the priority and experience
        """
        # add data to data and updates corresponding to the priority
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # update the sum tree with new priority
        self.update(tree_idx, priority)
        # increment the data pointer as data is stored from left to right nodes
        # create a ring buffer with fixed capacity by overwriting new data
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """
        update the priority
        """
        # change - the update factor for the sum tree
        change = priority - self.tree[tree_idx]
        # store the priority first in the tree
        self.tree[tree_idx] = priority
        # update the change to parent nodes
        self._propagate(tree_idx, change)

    def get(self, s):
        """
        get the leaf index, priority value of that leaf, and experience stored in that index
        """
        # traverse the tree from the top to bottom
        idx = self._retrieve(0, s)
        # the max priority which we could fetch for given input priority
        data_idx = idx - self.capacity + 1

        # return the index of priorities, priorities, and experiences
        return idx, self.tree[idx], self.data[data_idx]

    def _propagate(self, idx, change):
        """
        update to the root node, change the sum of priority value in all parent nodes
        """
        parent = (idx - 1) // 2
        # update priority of parent node
        self.tree[parent] += change

        if parent != 0:
            # end recursion if root node is the parent
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        find sample(experience) on leaf node
        """
        # idx: parent node beginning at root
        # s: uniform random priority within a segment

        # left and right child for idx
        left = 2 * idx + 1
        right = left + 1
        # base condition to break once tree traversal is complete(when there's no more child node)
        if left >= len(self.tree):
            return idx
        # find the node with maximum priority and traverse the subtree
        if s <= self.tree[left]:
            # move down to left of tree with parent node = left
            return self._retrieve(left, s)
        else:
            # move down to right of tree with parent node = right
            return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self):
        # find the sum of nodes by returning the root node
        return self.tree[0]

    @property
    # find the max priority (from the leaves)
    def max_p(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.capacity:]) / self.total_priority


class PrioritizedBuffer:
    """
    An implementation of proportional prioritized experience replay buffer
    """

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        # build a SumTree
        self.tree = SumTree(self.capacity)
        # clipped absolute error
        self.abs_error_upper = 1.0
        # a small positive constant that ensures that no transition has zero priority.
        self.epsilon = 0.01
        # control the tradeoff between sampling only experience with high priority and randomly, [0, 1]
        self.alpha = 0.6
        # importance sampling, from initial value incrementing to 1.
        self.beta = 0.4
        self.beta_increment = 0.001

        self.current_length = 0

    def save(self, state, legal_actions, action, reward, next_state, next_legal_actions, done):

        max_priority = self.tree.max_p
        # new experiences are first stored in the tree with max priority
        # untrained neural network is likely to return a value around zero for every input,
        # use a minimum priority to ensure every experience get a chance to be sampled
        if max_priority == 0:
            max_priority = self.abs_error_upper
        experience = (state, legal_actions, action, reward, next_state, next_legal_actions, done)
        self.tree.add(max_priority, experience)

    def sample(self):
        batch = []
        indices = np.empty((self.batch_size, ), dtype=np.int64)
        priorities = np.empty((self.batch_size, ), dtype=np.float32)

        # split the total priority into batch-sized segments
        priority_segment = self.tree.total_priority / self.batch_size
        # scheduler for beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(self.batch_size):
            # calculate priority value for each segment to get corresponding experiences
            a, b = priority_segment * i, priority_segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, experience = self.tree.get(s)

            # store the index and experience
            indices[i] = idx
            priorities[i] = priority

            batch.append(experience)

        # Priority is translated to probability of being chosen for replay. P(j)
        # A sample i has a probability of being picked during the experience replay determined by prob =pi / ∑kpk
        sampling_probs = priorities / self.tree.total_priority
        # weight update factor depends on sampling probability of the experience and beta
        # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        # normalize weights by 1/maxi wi for stability
        # keep all weights within a reasonable range, avoiding the possibility of extremely large updates
        is_weights /= np.max(is_weights)
        state_batch = []
        legal_actions_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_legal_actions_batch = []
        done_batch = []

        for transition in batch:
            state, legal_action, action, reward, next_state, next_legal_action, done = transition
            state_batch.append(state)
            legal_actions_batch.append(legal_action)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_legal_actions_batch.append(next_legal_action)
            done_batch.append(done)

        return state_batch, legal_actions_batch, action_batch, reward_batch, \
               next_state_batch, next_legal_actions_batch, done_batch, indices, is_weights

    def update_priorities(self, tree_idx, abs_error):
        abs_error += self.epsilon
        clipped_errors = np.minimum(abs_error, self.abs_error_upper)
        # The error is converted to priority by p=(error+ϵ)**α
        priorities = np.power(clipped_errors, self.alpha)

        for tree_index, priority in zip(tree_idx, priorities):
            self.tree.update(tree_index, priority)

    def __len__(self):
        return self.current_length


class NStepPERBuffer(PrioritizedBuffer):

    def __init__(self, capacity, batch_size, n_step, gamma):
        super().__init__(capacity, batch_size)
        self.capacity = capacity
        self.batch_size = batch_size
        # build a SumTree
        self.tree = SumTree(self.capacity)
        # clipped absolute error
        self.abs_error_upper = 1.0
        # a small positive constant that ensures that no transition has zero priority.
        self.epsilon = 0.01
        # control the tradeoff between sampling only experience with high priority and randomly
        self.alpha = 0.6
        # importance sampling, from initial value incrementing to 1.
        self.beta = 0.4
        self.beta_increment = 0.001
        self.current_length = 0

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_state, next_legal_actions, done = self.n_step_buffer[-1][-4:]

        for _, _, _, reward_, next_state_, _, done_ in reversed(list(self.n_step_buffer)[: -1]):
            reward = reward_ + self.gamma * reward * (1 - done_)
            next_state, done = (next_state_, done_) if done_ else (next_state, done)
        return reward, next_state, next_legal_actions, done

    def save(self, state, legal_actions, action, reward, next_state, next_legal_actions, done):

        max_priority = self.tree.max_p
        if max_priority == 0:
            max_priority = self.abs_error_upper

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.n_step_buffer.append([state, legal_actions, action, reward, next_state, next_legal_actions, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, next_legal_actions, done = self._get_n_step_info()
        state, legal_actions, action = self.n_step_buffer[0][: 3]
        experience = (state, legal_actions, action, reward, next_state, next_legal_actions, done)
        self.tree.add(max_priority, experience)

