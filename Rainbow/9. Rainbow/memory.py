from collections import namedtuple, deque

import numpy as np

from config import *
from model import *

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class MemoryWithTDError(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        if len(self.memory) > 0:
            max_probability = max(self.memory_probability)
        else:
            max_probability = e_adj
        self.memory.append(Transition(state, next_state, action, reward, done))
        self.memory_probability.append(max_probability)

    # noinspection PyShadowingNames
    def sample(self, batch_size, live_net, target_net, beta):
        probability_sum = sum(self.memory_probability)
        p = [probability / probability_sum for probability in self.memory_probability]

        indices = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)

        transitions = [self.memory[idx] for idx in indices]
        transitions_p = [p[idx] for idx in indices]

        batch = Transition(*zip(*transitions))

        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.tensor(weights).to(device)
        weights = weights / weights.max()

        td_error = DQN.get_td_error(live_net, target_net,
                                    batch.state, batch.next_state, batch.action, batch.reward, batch.done)

        for idx, single_td_error in zip(indices, td_error):
            self.memory_probability[idx] = pow(abs(single_td_error) + e_adj, alpha).item()

        return batch, weights

    def __len__(self):
        return len(self.memory)
