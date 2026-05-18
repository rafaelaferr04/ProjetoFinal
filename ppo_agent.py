import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorCritic
from rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        num_states,
        num_actions,
        gamma,
        learning_rate,
        clip_epsilon,
        ppo_epochs,
        hidden_size
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

        self.model = ActorCritic(num_states, num_actions, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.buffer = RolloutBuffer()

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_logits, state_value = self.model(state_tensor)
            distribution = torch.distributions.Categorical(logits=action_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        return action.item(), log_prob, state_value.squeeze()

    def choose_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_logits, _ = self.model(state_tensor)

        return torch.argmax(action_logits).item()

    def calculate_gae(self, last_value=0.0, gae_lambda=0.95):
        advantages = []
        gae = 0

        values = self.buffer.values + [last_value]

        for step in reversed(range(len(self.buffer.rewards))):
            delta = (
                self.buffer.rewards[step]
                + self.gamma * values[step + 1] * (1 - self.buffer.dones[step])
                - values[step]
            )

            gae = delta + self.gamma * gae_lambda * (1 - self.buffer.dones[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        values = torch.tensor(self.buffer.values, dtype=torch.float32)
        returns = advantages + values

        return returns, advantages

    def update(self):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()

        returns, advantages = self.calculate_gae()
        returns = returns.detach()
        advantages = advantages.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            action_logits, state_values = self.model(states)
            distribution = torch.distributions.Categorical(logits=action_logits)

            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            state_values = state_values.squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * advantages

            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.clear()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()