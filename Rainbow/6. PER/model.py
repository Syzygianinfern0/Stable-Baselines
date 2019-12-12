import torch
import torch.nn as nn

from config import gamma, device


# noinspection PyPep8Naming


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
