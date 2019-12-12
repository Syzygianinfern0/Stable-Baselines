import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma


class DQN(nn.Module):
    def __init__(self, no_states, no_actions):
        super(DQN, self).__init__()
        self.num_inputs = no_states
        self.num_outputs = no_actions

        self.net = nn.Sequential(
            nn.Linear(no_states, 128),
            nn.ReLU(),
            nn.Linear(128, no_actions)
        )

    def forward(self, state):
        qvalue = self.net(state)
        return qvalue

    @classmethod
    def train_model(cls, net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.FloatTensor(batch.action)
        rewards = torch.Tensor(batch.reward)
        dones = torch.Tensor(batch.dones)

        current_q_value = net(states).squeeze(1)
        next_pred = net(next_states).squeeze(1)

        pred = torch.sum(current_q_value.mul(actions), dim=1)

        target = rewards + (1 - dones) * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # noinspection PyShadowingBuiltins
    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
