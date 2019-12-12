import math

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from config import gamma, device, sigma_zero


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        # noinspection PyUnresolvedReferences
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    # noinspection PyShadowingBuiltins
    def forward(self, input: torch.Tensor):
        return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class DQN(nn.Module):
    def __init__(self, no_states, no_actions):
        super(DQN, self).__init__()
        self.num_inputs = no_states
        self.num_outputs = no_actions

        self.fc = nn.Linear(no_states, 128)
        self.fc_adv = NoisyLinear(128, no_actions)
        self.fc_val = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))

        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)

        val = self.fc_val(x)
        val = val.view(-1, 1)

        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))
        return qvalue

    # noinspection PyArgumentList
    @classmethod
    def get_td_error(cls, live_net, target_net, states, next_states, actions, rewards, dones):
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        dones = torch.Tensor(dones).to(device)

        current_q_value = live_net(states).squeeze(1)
        pred = torch.sum(current_q_value.mul(actions), dim=1)

        next_pred = target_net(next_states).squeeze(1)

        # noinspection PyTypeChecker
        target = rewards + (1 - dones) * gamma * next_pred.max(1)[0]

        td_error = pred - target.detach()

        return td_error

    # noinspection PyArgumentList
    @classmethod
    def train_model(cls, live_net, target_net, optimizer, batch, weights):
        td_error = cls.get_td_error(live_net, target_net,
                                    batch.state, batch.next_state, batch.action, batch.reward, batch.done)

        loss = torch.pow(td_error, 2) * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # noinspection PyShadowingBuiltins
    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.item()

    def reset_noise(self):
        self.fc2.reset_noise()
