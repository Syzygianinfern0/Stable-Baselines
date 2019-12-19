import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from config import gamma


class ActorCritic(nn.Module):
    def __init__(self, no_states, no_actions):
        super(ActorCritic, self).__init__()
        self.no_states = no_states
        self.no_actions = no_actions

        self.fc = nn.Linear(no_states, 128)
        self.fc_actor = nn.Linear(128, no_actions)
        self.fc_critic = nn.Linear(128, no_actions)

    def forward(self, state):
        x = F.relu(self.fc(state))
        policy = F.softmax(self.fc_actor(x))
        q_value = self.fc_critic(x)
        return policy, q_value

    @classmethod
    def train_model(cls, net, optimizer, transition):
        state, next_state, action, reward, done = transition

        policy, q_value = net(state)
        policy, q_value = policy.squeeze(), q_value.squeeze()

        _, next_q_value = net(next_state)
        next_q_value = next_q_value.squeeze()

        next_action = net.get_action(next_state)

        target = reward + (1 - done) * gamma * next_q_value[next_action]

        log_policy = torch.log(policy)[action]
        loss_policy = log_policy * q_value[action].item()
        loss_value = F.smooth_l1_loss(q_value[action], target.detach())

        loss = loss_value + loss_policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        policy, _ = self.forward(state)
        policy = policy.squeeze().cpu().detach().numpy()

        action = np.random.choice(self.no_actions, 1, p=policy)[0]
        return action
