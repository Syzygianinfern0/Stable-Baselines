import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.distributions import Categorical

from config import device, gamma


class GAE(nn.Module):
    def __init__(self, no_states, no_actions):
        super(GAE, self).__init__()
        self.no_states = no_states
        self.no_actions = no_actions

        self.policy = nn.Sequential(
            nn.Linear(no_states, 32),
            nn.ReLU(),
            nn.Linear(32, no_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(no_states, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        policy_logits = self.policy(state)
        value = self.value(state)
        return policy_logits, value

    @classmethod
    def train_model(cls, net, optimizer, trajectory, rtg=True):
        states = torch.stack(trajectory.state).to(device)
        actions = torch.tensor(trajectory.action).to(device)
        # rewards = torch.tensor(trajectory.reward).to(device)
        rewards = trajectory.reward

        logits = net(states).squeeze()
        log_probs = F.log_softmax(logits)
        sum_log_probs = torch.sum(log_probs * actions, dim=1)

        if rtg:
            weights = lfilter([1], [1, float(-gamma)], list(rewards)[::-1], axis=0)[::-1]
        else:
            weights = torch.sum(rewards)
        g = sum_log_probs * torch.tensor(weights.copy()).to(device)

        loss = -1 * torch.mean(g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        logits = self.forward(state).squeeze()

        distribution = Categorical(logits=logits)
        action = distribution.sample().item()

        return action
