from collections import deque

import gym
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import *
from memory import Memory
from model import REINFORCE


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net: REINFORCE = REINFORCE(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()

    running_score = deque(maxlen=100)
    scores = []
    steps = 0
    loss = 0

    for e in range(max_episodes):
        done = False
        memory = Memory()

        score = 0
        state = env.reset()
        state = torch.tensor(state).float().to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.tensor(next_state).float().to(device)
            next_state = next_state.unsqueeze(0)

            # mask = 0 if done else 1
            # reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(num_actions)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, done)

            score += reward
            state = next_state

        loss = REINFORCE.train_model(net, memory.sample(), optimizer)

        # score = score if score == 500.0 else score + 1
        # running_score = 0.99 * running_score + 0.01 * score
        writer.add_scalar('log/score', score, e)
        scores.append(score)
        running_score.append(score)

        if e % log_interval == 0:
            print(f'{e} episode | score: {np.mean(running_score):.2f}')
            writer.add_scalar('log/avg', np.mean(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if np.mean(running_score) > goal_score:
            writer.add_scalar('log/avg', np.mean(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)
            print(f'{env_name} solved in {e} episodes!!')
            torch.save(net.net.state_dict(), 'trained.pth')
            break

    writer.close()


if __name__ == "__main__":
    main()
