import gym
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import *
from memory import *
from model import *


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


def main():
    print(f"Training on {device}")
    print()
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    live_net: DQN = DQN(num_inputs, num_actions)
    target_net: DQN = DQN(num_inputs, num_actions)
    update_target_model(live_net, target_net)

    optimizer = optim.Adam(live_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    live_net.to(device)
    live_net.train()

    target_net.train()
    target_net.to(device)

    memory = MemoryWithTDError(experience_size)
    running_score = deque(maxlen=100)
    scores = []
    steps = 0
    loss = 0
    beta = min_beta

    for e in range(max_episodes):
        done = False

        score = 0
        state = env.reset()
        state = torch.tensor(state).float().to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = live_net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.tensor(next_state).float().to(device)
            next_state = next_state.unsqueeze(0)

            action_one_hot = np.zeros(num_actions)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, done)

            score += reward
            state = next_state

            if steps > initial_exploration:
                beta = max_beta - (max_beta - min_beta) * np.exp(-1 * e * beta_anneal)

                batch, weights = memory.sample(batch_size, live_net, target_net, beta)
                loss = DQN.train_model(live_net, target_net, optimizer, batch, weights)

                if steps % update_target == 0:
                    update_target_model(live_net, target_net)

        writer.add_scalar('log/score', score, e)
        scores.append(score)
        running_score.append(score)

        if e % log_interval == 0:
            print(f'{e} episode | score: {np.mean(running_score):.2f} | beta: {beta:.2f}')
            writer.add_scalar('log/avg', np.mean(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if np.mean(running_score) > goal_score:
            writer.add_scalar('log/avg', np.mean(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)
            print(f'{env_name} solved in {e} episodes!!')
            torch.save(live_net.state_dict(), 'trained.pth')
            break


if __name__ == "__main__":
    main()
