import torch
import random
import numpy as np
from collections import deque, namedtuple

MDPTuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

"""
    Both SegmentTree and PrioritizedMemory classes were inspired by the
    Deep Mind repository file https://github.com/deepmind/dqn_zoo/blob/f011d683529d8d23b017a95194ebbb41a4962fe8/dqn_zoo/replay.py#L305
"""

class ReplayMemory():
    def __init__(self, memory_size, batch_size, device, seed=0):
        # Saving variables
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._device = device
        self._random_seed = random.seed(seed)
        self._numpy_seed = np.random.seed(seed)

        # Memory variables
        self._memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self._memory)

    def save(self, state, action, reward, next_state, done):
        """ Save new experience """

        self._memory.append(MDPTuple(state, action, reward, next_state, done))

    def sample(self):
        """ Sample a batch of random experiences from memory """

        # Get memory tuples
        tuples = random.sample(self._memory, k=self._batch_size)
        # Create torch tensors from data
        states = torch.from_numpy(np.vstack([e.state for e in tuples if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e.action for e in tuples if e is not None])).float().to(self._device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in tuples if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in tuples if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e.done for e in tuples if e is not None]).astype(np.uint8)).float().to(self._device)
        return (states, actions, rewards, next_states, dones)


class SegmentTree():
    def __init__(self, memory_size, tree_type='sum'):
        # Input variables
        self._type = tree_type
        self._memory_size = memory_size

        # Get tree length
        self._length = 1
        while self._length < memory_size:
            self._length *= 2

        self._total_length = self._length + memory_size

        # Structure
        self._tree = np.zeros((self._total_length,))
        self._index = 0
        self._max_value = 1

    # ---------------  Update functions  ---------------
    def update(self, indices, values):
        """ Update leafs with new values and propagate through parent nodes """

        indices = indices + self._length
        # Update leafs
        self._tree[indices] = values
        # Propagate throw all affected nodes
        self.multi_propagate(indices)
        # Get new max value
        batch_max = max(values)
        self._max_value = max(self._max_value, batch_max)

    def update_nodes(self, indices):
        """ Update parent nodes with changes """

        # Get children indices
        children_indices = np.array([indices * 2, (indices * 2) + 1])
        children_indices[children_indices > self._total_length - 1] = 0
        # Update parents with children sums
        self._tree[indices] = np.sum(self._tree[children_indices], axis=0)

    # -------------  Propagete functions  --------------
    def propagate(self, index, diff):
        """ Simple propagation when single leaf changes """

        # Get parent of indexed node
        parent_index = index // 2
        # Add difference
        self._tree[parent_index] += diff
        # Propagate until root
        if parent_index != 1:
            self.propagate(parent_index, diff)

    def multi_propagate(self, indices):
        """ Propagation for many leaf changes """

        # Get parent nodes
        parent_indices = np.unique(indices // 2)
        # Get parent
        self.update_nodes(parent_indices)
        # Propagate until root
        if parent_indices[0] != 1:
            self.multi_propagate(parent_indices)

    # -----------------  Save function  ----------------
    def append_max(self):
        """ Append max priority in corresponding index """

        # Compute tree index
        index = self._length + self._index
        # Get difference between the new value and the old one
        diff = self._max_value - self._tree[index]
        # Update tree with new value
        self._tree[index] = self._max_value
        # Propagate difference
        if diff != 0: self.propagate(index, diff)
        # Update index
        self._index = (self._index + 1) % self._memory_size

    # ---------------  Search functions  ---------------
    def retrieve(self, indices, values):
        """ Find smallest indices for a list of values """

        # Get children indices
        children_indices = np.array([indices * 2, (indices * 2) + 1])
        # Return leafs
        if children_indices[0, 0] >= self._total_length:
            return indices
        # Checking if children are leaf, avoiding overflowed leafs
        elif children_indices[0, 0] >= self._length:
            children_indices = np.minimum(children_indices, self._total_length - 1)

        # Get all left branch values
        left_branch_values = self._tree[children_indices[0]]
        # If node value is smaller than value, stop searching in that way
        indices_mask = np.greater(values, left_branch_values).astype(np.int32)
        # Get forward indices and values
        forward_indices = children_indices[indices_mask, np.arange(indices.size)]
        forward_values = values - (indices_mask * left_branch_values)
        # Keep iterating
        return self.retrieve(forward_indices, forward_values)

    def find(self, values):
        """ Search for indices given a set of values """

        indices = self.retrieve(np.ones(values.shape, dtype=np.int32), values)
        real_indices = indices - self._length
        return self._tree[indices], real_indices

    def get_root(self):
        """ Get tree root, which represents the sum of all priorities """
        return self._tree[1]


class PrioritizedMemory():
    def __init__(self,
                 memory_size,
                 batch_size,
                 device,
                 seed=0,
                 prioritization=0.5,
                 correction=0.4,
                 correction_increase_rate=7e-6,
                 offset=0.01):

        # Saving variables
        self._memory_size = memory_size                              #  Max number of MDP tuples
        self._batch_size = batch_size                                #  Samples per batch
        self._device = device

        # Memory variables
        self._memory = deque(maxlen=memory_size)                     #  Main memory to store MDP tuples
        self._sum_tree = SegmentTree(memory_size)                    #  Segment tree for optimal search

        # Hyperparameters
        self._prioritization = prioritization                        #  Prioritization exponent
        self._correction = correction                                #  Weight correction
        self._correction_increase_rate = correction_increase_rate    #  Increase rate for weight correction
        self._offset = offset                                        #  Offset value to avoid 0-valued priorities

        # Set seeds
        self._random_seed = random.seed(seed)
        self._numpy_seed = np.random.seed(seed)

    def __len__(self):
        return len(self._memory)

    def save(self, state, action, reward, next_state, done):
        """ Save new experience with its priority """

        self._memory.append(MDPTuple(state, action, reward, next_state, done))
        self._sum_tree.append_max()

    def get_proportional_sample(self):
        """ Get uniform samles throw segmentation of the distribution """

        priorities_sum = self._sum_tree.get_root()
        segment_len = priorities_sum / self._batch_size
        segment_starts = np.arange(self._batch_size) * segment_len

        while True:
            # Uniformly sample from segment
            values = np.random.uniform(0.0, segment_len, [self._batch_size]) + segment_starts
            probs, indices = self._sum_tree.find(values)
            # All probabilities must be greater than zero
            if np.all(probs != 0): break

        # Get appropriate values for probs
        probs = probs / priorities_sum
        return indices, np.array(probs)

    def get_experiences(self, indices):
        """ Get experience tuples from memory """

        tuples = np.array([ self._memory[i] for i in indices ])
        states, actions, rewards, next_states, dones = zip(*tuples)

        # Convert to numpy array
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

    def get_weights(self, probs):
        """ Compute weights for priorities' relevance in the learning step """

        weights = (len(self._memory) * probs) ** -self._correction
        norm_weights = weights / weights.max()
        return norm_weights

    def sample(self):
        """ Sample a batch of experiences from memory """

        # Search for indices and probs in sum tree
        indices, probs = self.get_proportional_sample()
        # Get data from experiences
        states, actions, rewards, next_states, dones = self.get_experiences(indices)
        # Compute weights
        weights = self.get_weights(probs)

        # Create torch tensors
        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).float().to(self._device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self._device)
        next_states = torch.from_numpy(next_states).float().to(self._device)
        dones = torch.from_numpy(dones.astype(np.uint8)).unsqueeze(1).float().to(self._device)
        weights = torch.from_numpy(weights).unsqueeze(1).float().to(self._device)

        return (states, actions, rewards, next_states, dones), weights, indices

    def update_priorities(self, indices, errors):
        """ Update batch priorities with new Temporal-Difference errors """

        priorities = np.power(errors + self._offset, self._prioritization)
        self._sum_tree.update(indices, priorities)
        self._correction += self._correction_increase_rate
