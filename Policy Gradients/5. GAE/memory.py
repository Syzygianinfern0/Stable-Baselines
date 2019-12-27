import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done', 'value'))


class Memory(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, done, value):
        self.memory.append(Transition(state, next_state, action, reward, done, value))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
