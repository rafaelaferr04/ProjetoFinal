import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size=128):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_size, num_actions)

        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.shared(state)

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value