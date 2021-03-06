import numpy as np
import torch.nn as nn

from config import *


class REINFORCE(nn.Module):
    def __init__(self, no_states, no_actions):
        super(REINFORCE, self).__init__()
        self.no_states = no_states
        self.no_actions = no_actions

        self.net = nn.Sequential(
            nn.Linear(no_states, 128),
            nn.ReLU(),
            nn.Linear(128, no_actions),
            nn.Softmax()
        )

    def forward(self, state):
        policy = self.net(state)
        return policy

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, dones = transitions.state, transitions.action, transitions.reward, transitions.done

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)

        returns = torch.zeros_like(rewards)

        running_return = 0
        # noinspection PyTypeChecker
        for t in reversed(range(len(rewards))):
            # noinspection PyTypeChecker
            running_return = rewards[t] + gamma * running_return * ~dones[t]
            returns[t] = running_return

        policies = net(states)
        policies = policies.view(-1, net.no_actions)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)

        loss = (-log_policies * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        policy = self.forward(state)
        policy = policy.squeeze().cpu().detach().numpy()

        action = np.random.choice(self.no_actions, 1, p=policy)[0]
        return action
