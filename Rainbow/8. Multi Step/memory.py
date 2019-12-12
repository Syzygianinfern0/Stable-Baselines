import random
from collections import namedtuple, deque
from config import n_step, gamma

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

        self.local_step = 0
        self.local_state = None
        self.local_action = None
        self.local_rewards = []

    def reset_local(self):
        self.local_step = 0
        self.local_state = None
        self.local_action = None
        self.local_rewards = []

    def push(self, state, next_state, action, reward, done):
        self.local_step += 1
        self.local_rewards.append(reward)

        if self.local_step == 1:
            self.local_state = state
            self.local_action = action

        if self.local_step == n_step:
            reward = 0
            for idx, local_reward in enumerate(self.local_rewards):
                reward += (gamma ** idx) * local_reward
            self.memory.append(Transition(self.local_state, next_state, self.local_action, reward, done))
            self.reset_local()
        if done == 1:
            self.reset_local()

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
