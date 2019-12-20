from collections import deque

import gym
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import *
from model import ActorCritic


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net: ActorCritic = ActorCritic(num_inputs, num_actions)

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

            transition = [state, next_state, action, reward, done]

            score += reward
            state = next_state

            loss = ActorCritic.train_model(net, optimizer, transition)

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
            torch.save(net.state_dict(), 'trained.pth')
            break

    writer.close()


if __name__ == "__main__":
    main()
