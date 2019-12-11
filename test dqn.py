# %%

import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib
from collections import deque
from torch import optim
import random
from torch.autograd import Variable

# %%

env = gym.make('CartPole-v1')

# %%

print(env.action_space)
print(env.observation_space)

no_actions = env.action_space.n
no_observations = env.observation_space.shape[0]

# %%

MAX_EPISODES = 1000
EXPERIENCE_SIZE = 1000000
BATCH_SIZE = 24
GAMMA = 1.0
MAX_EPSILON = EPSILON = 1.0
MIN_EPSILON = 1e-2
DECAY_RATE = 0.005
LEARNING_RATE = 2e-2


# %%

# noinspection PyShadowingNames
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.replay_memory = deque(maxlen=EXPERIENCE_SIZE)

        self.net = nn.Sequential(
            nn.Linear(no_observations, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, no_actions)
        )
        self.net = self.net.cuda()
        self.optim = optim.Adam(self.net.parameters())

    def act(self, state):
        if np.random.uniform(0, 1) > EPSILON:
            q_values = self.net(torch.tensor(state).unsqueeze(0).cuda())
            action = torch.argmax(q_values.squeeze()).item()
        else:
            action = env.action_space.sample()
        return action

    def learn(self, episode):
        if len(self.replay_memory) < BATCH_SIZE:
            return
        train_batch = random.sample(self.replay_memory, BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*train_batch)

        batch_state = torch.tensor(batch_state).float().cuda()
        batch_next_state = torch.tensor(batch_next_state).float().cuda()
        batch_action = torch.tensor(batch_action).cuda().unsqueeze(1)
        batch_done = torch.tensor(batch_done).cuda().to(dtype=torch.int)
        batch_reward = torch.tensor(batch_reward).cuda().to(dtype=torch.int)

        current_q_values = self.net(batch_state).gather(1, batch_action)
        target_q_values = batch_reward + (torch.ones_like(batch_done) - batch_done) * (GAMMA * self.net(batch_next_state).max(1)[0])
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # noinspection PyPep8Naming,PyUnusedLocal
        EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-1 * DECAY_RATE * episode)

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))


dqn = DQN()

# %%

rewards = []
rewards_dq = deque(maxlen=100)

for episode in range(MAX_EPISODES):
    # noinspection PyRedeclaration
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = dqn.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        dqn.remember(state, action, reward, next_state, done)
        dqn.learn(episode)
        state = next_state

    rewards.append(total_reward)
    rewards_dq.append(total_reward)

    if not episode % 30:
        print(f'Episode : {episode}')
        print(f'Best Reward : {max(rewards)}')
        print(f'Mean over last 50 : {np.mean(rewards_dq)}')
        print(f'Epsilon : {EPSILON}')
        print()
        if np.mean(rewards_dq) > 195:
            break
