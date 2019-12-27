import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.distributions import Categorical

from config import device, gamma, lmbda


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
    def train_model(cls, net, optimizer, batch, rtg=True):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(batch.action).to(device)
        rewards = torch.tensor(batch.reward).to(device)
        dones = torch.tensor(batch.done).to(device)
        values = torch.tensor(batch.value).to(device)

        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        advantages = lfilter([1], [1, float(-gamma * lmbda)], list(deltas)[::-1], axis=0)[::-1]
        returns = lfilter([1], [1, float(-gamma)], list(rewards)[::-1], axis=0)[::-1]

        advantages, returns = torch.tensor(advantages).to(device), torch.tensor(returns).to(device)

        mu, dev = advantages.mean(), advantages.std()
        advantages = (advantages - mu) / dev

        logits, off_values = net(states).squeeze()

        log_probs = F.log_softmax(logits)
        sum_log_probs = torch.sum(log_probs * actions, dim=1)
        policy_loss = -1 * torch.mean(sum_log_probs * advantages)

        value_loss = F.mse_loss(returns, off_values)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state):
        logits, value = self.forward(state).squeeze()

        distribution = Categorical(logits=logits)
        action = distribution.sample().item()

        return action, value
