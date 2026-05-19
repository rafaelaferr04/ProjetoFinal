import numpy as np
import pickle

from fourier_basis import FourierBasis
from config import LEARNING_RATE, GAMMA, LAM, ORDER, START_EPSILON, FINAL_EPSILON, EPSILON_DECAY

# Desconto por tempo
_STEP_PENALTY = 0.1        # -0.1 por passo

# Avaliação no primeiro toque (ambas as pernas)
_TOUCH_BONUS      = 100.0  # toque dentro das bandeiras
_TOUCH_PENALTY    = 80.0   # penalização máx fora (proporcional à distância)

# Avaliação final (lander completamente parado, terminated=True)
_FINAL_BONUS      = 150.0  # parou dentro das bandeiras
_FINAL_PENALTY    = 120.0  # parou fora das bandeiras


def _touchdown_reward(obs):
    """Bónus/penalização no momento em que ambas as pernas tocam pela primeira vez."""
    x_pos = float(obs[0])
    if abs(x_pos) <= 0.15:
        return _TOUCH_BONUS
    else:
        dist_frac = min(1.0, (abs(x_pos) - 0.15) / 0.85)
        return -_TOUCH_PENALTY * dist_frac


def _final_reward(obs):
    """Bónus/penalização quando o lander para completamente (terminated=True)."""
    left_leg  = float(obs[6])
    right_leg = float(obs[7])
    if left_leg > 0.5 and right_leg > 0.5:
        x_pos = float(obs[0])
        if abs(x_pos) <= 0.15:
            return _FINAL_BONUS
        else:
            return -_FINAL_PENALTY
    return 0.0


class SarsaLambdaAgent:

    def __init__(self, env, order=ORDER):
        self.env         = env
        self.state_dim   = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        low  = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        self.low  = np.where(np.isfinite(low),  low,  -1.0)
        self.high = np.where(np.isfinite(high), high,  1.0)

        self.basis        = FourierBasis(self.state_dim, order)
        self.num_features = self.basis.num_features

        self.alpha   = LEARNING_RATE
        self.gamma   = GAMMA
        self.lam     = LAM

        self.epsilon       = START_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.final_epsilon = FINAL_EPSILON

        norms = np.maximum(1.0, np.linalg.norm(self.basis.c, axis=1))
        self.alpha_vec = self.alpha / norms

        self.weights = np.zeros((self.num_actions, self.num_features))
        self.training_error = []

    def normalize_state(self, state):
        return np.clip((state - self.low) / (self.high - self.low), 0.0, 1.0)

    def get_q(self, state):
        features = self.basis.get_features(self.normalize_state(state))
        return np.dot(self.weights, features), features

    def choose_action(self, state):
        # Bloqueia propulsores quando ambas as pernas estão no chão
        if state[6] > 0.5 and state[7] > 0.5:
            return 0
        q_values, _ = self.get_q(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(q_values)

    def train_episode(self, obs):
        action             = self.choose_action(obs)
        q_values, features = self.get_q(obs)
        e = np.zeros((self.num_actions, self.num_features))
        episode_reward = 0.0
        done   = False
        landed = False

        while not done:
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Desconto por tempo — penaliza episódios longos
            reward -= _STEP_PENALTY

            # 1ª avaliação: primeiro toque com ambas as pernas
            if not landed and next_obs[6] > 0.5 and next_obs[7] > 0.5:
                landed = True
                reward += _touchdown_reward(next_obs)

            # 2ª avaliação: posição final após qualquer deslize (lander parado)
            if terminated:
                reward += _final_reward(next_obs)

            episode_reward += reward

            next_action               = self.choose_action(next_obs)
            next_q_values, next_feats = self.get_q(next_obs)

            if done:
                delta = reward - q_values[action]
            else:
                delta = (
                    reward
                    + self.gamma * next_q_values[next_action]
                    - q_values[action]
                )

            delta = float(np.clip(delta, -50.0, 50.0))

            e        *= self.gamma * self.lam
            e[action]  = features

            self.weights += self.alpha_vec * delta * e
            np.clip(self.weights, -200.0, 200.0, out=self.weights)

            self.training_error.append(delta)
            obs, action, q_values, features = (
                next_obs, next_action, next_q_values, next_feats
            )

        return episode_reward

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.weights,
                "order":   self.basis.order,
                "low":     self.low,
                "high":    self.high,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        if "low" in data:
            self.low  = data["low"]
            self.high = data["high"]
        self.epsilon = 0.0