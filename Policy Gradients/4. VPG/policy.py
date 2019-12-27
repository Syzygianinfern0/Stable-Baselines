import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.distributions import Categorical

from config import device, gamma


class VPG(nn.Module):
    def __init__(self, no_states, no_actions):
        super(VPG, self).__init__()
        self.no_states = no_states
        self.no_actions = no_actions

        self.net = nn.Sequential(
            nn.Linear(no_states, 32),
            nn.Tanh(),
            nn.Linear(32, no_actions),
        )

    def forward(self, state):
        logits = self.net(state)
        return logits

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
        # probs = F.softmax(logits)
        # probs = probs.squeeze().cpu().detach().numpy()

        # action = np.random.choice(self.no_actions, 1, p=probs)[0]
        return action
