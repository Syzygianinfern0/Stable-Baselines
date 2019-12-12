import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from config import gamma, device


class DQN(nn.Module):
    def __init__(self, no_states, no_actions):
        super(DQN, self).__init__()
        self.num_inputs = no_states
        self.num_outputs = no_actions

        self.fc = nn.Linear(no_states, 128)
        self.fc_adv = nn.Linear(128, no_actions)
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
    def train_model(cls, live_net, target_net, optimizer, batch):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.FloatTensor(batch.action).to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        dones = torch.Tensor(batch.done).to(device)

        current_q_value = live_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(current_q_value.mul(actions), dim=1)

        # noinspection PyTypeChecker
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
        return action.item()
