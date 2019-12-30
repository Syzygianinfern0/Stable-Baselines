import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.distributions import Categorical
from torch.nn.utils import parameters_to_vector

from config import device, gamma, lmbda, cg_iters, eps


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
    def train_model(cls, net, optimizer, batch):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(batch.action).to(device)
        rewards = torch.tensor(batch.reward).to(device)
        dones = torch.tensor(batch.done).to(device)
        values = torch.tensor(batch.value).to(device)

        # if dones[-1] == True:
        #     torch.cat([rewards, 0], )

        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

        advantages = cls.discounted_cum_sum(deltas, gamma * lmbda)
        returns = cls.discounted_cum_sum(rewards, gamma)

        # advantages, returns = torch.tensor(advantages.copy()).to(device), torch.tensor(returns.copy()).to(device)

        mu, dev = advantages.mean(), advantages.std()
        advantages = (advantages - mu) / dev

        logits, _ = net(states)
        # logits, off_values = logits.squeeze(), off_values.squeeze()

        log_probs = F.log_softmax(logits.squeeze())
        sum_log_probs = torch.sum(log_probs * actions, dim=1)
        policy_loss = -1 * torch.mean(sum_log_probs[:-1] * advantages)

        value_loss = F.mse_loss(returns, values)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, policy_loss, value_loss

    def get_action(self, state):
        logits, value = self.forward(state)
        logits, value = logits.squeeze(), value.squeeze()

        distribution = Categorical(logits=logits)
        action = distribution.sample().item()

        return action, value

    @staticmethod
    def discounted_cum_sum(x, discount):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        cum_sum = lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        return torch.tensor(cum_sum.copy()).float().to(device)

    @staticmethod
    def conjucate_gradient(Ax, b):
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + eps)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    @classmethod
    def hessian_vector_product(cls, D_kl, module, x, retain=True):
        gradients = torch.autograd.grad(
            D_kl, module.parameters(),
            create_graph=True
        )

        flat_gradients = cls.flat_grad(gradients)
        flat_gradients = (flat_gradients * x).sum()

        hessian = torch.autograd.grad(
            flat_gradients, module.parameters(),
            retain_graph=retain
        )

        return cls.flat_grad(hessian)

    @staticmethod
    def flat_grad(grad):
        return parameters_to_vector(grad)