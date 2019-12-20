import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma


class ActorCritic(nn.Module):
    def __init__(self, no_states, no_actions):
        super(ActorCritic, self).__init__()
        self.no_states = no_states
        self.no_actions = no_actions

        self.fc = nn.Linear(no_states, 128)
        self.fc_actor = nn.Linear(128, no_actions)
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        policy = F.softmax(self.fc_actor(x))
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def train_model(cls, net, optimizer, transition):
        state, next_state, action, reward, done = transition

        policy, value = net(state)
        policy, value = policy.squeeze(), value.squeeze()
        log_policy = torch.log(policy)[action]

        _, next_value = net(next_state)
        next_value = next_value.squeeze()

        target = reward + (1 - done) * gamma * next_value
        td_error = target - value
        loss_policy = - log_policy * td_error

        loss_value = F.mse_loss(value, target)
        entropy = torch.log(policy[0]) * policy[0]

        # loss = loss_policy + loss_value - 0.1 * entropy.sum()
        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        policy, _ = self.forward(state)
        policy = policy.squeeze().cpu().detach().numpy()

        action = np.random.choice(self.no_actions, 1, p=policy)[0]
        return action
