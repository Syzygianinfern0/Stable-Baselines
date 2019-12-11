# %%

import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

# %%

env = gym.make('CartPole-v1')

# %%

print(env.action_space)
print(env.observation_space)

no_actions = env.action_space.n
no_observations = env.observation_space.shape[0]

# %%

MAX_EPISODES = 1000
EXPERIENCE_SIZE = 1_00_000
BATCH_SIZE = 64
GAMMA = 1.0
MAX_EPSILON = EPSILON = 1.0
MIN_EPSILON = 1e-2
DECAY_RATE = 0.005
LEARNING_RATE = 1e-2


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
        self.optim = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def act(self, state, test=False):
        if (np.random.uniform(0, 1) > EPSILON) or test:
            q_values = self.net(torch.tensor(state).float().unsqueeze(0).cuda())
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
        target_q_values = batch_reward + (GAMMA * self.net(batch_next_state).max(1)[0])
        # target_q_values = batch_reward + \
        #                   (torch.ones_like(batch_done) - batch_done) * (GAMMA * self.net(batch_next_state).max(1)[0])

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        if not episode % 30:
            print(loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))


dqn = DQN()


# %%
# noinspection PyUnusedLocal,PyShadowingNames
def test():
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = dqn.acTat(state, True)
        next_state, reward, done, _ = env.step(action)
        env.render()
        #         time.sleep(0.07)
        total_reward += reward
        state = next_state
    env.close()
    print(f"Evaluation Score : {total_reward}")
    print()


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
        EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-1 * DECAY_RATE * episode)
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
        test()

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
        EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-1 * DECAY_RATE * episode)
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
        test()
env.close()
