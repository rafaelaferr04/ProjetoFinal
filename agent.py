import numpy as np
import pickle

from fourier_basis import FourierBasis
from config import LEARNING_RATE, GAMMA, LAM, ORDER, START_EPSILON, FINAL_EPSILON, EPSILON_DECAY

_STEP_BONUS   =  0.2    # +0.2/passo reduz penalização de combustível
_BONUS_CENTER =  200.0  # |x| ≤ 0.2 — entre as bandeiras
_BONUS_NEAR   =   50.0  # 0.2 < |x| ≤ 0.5 — perto das bandeiras
_PENALTY_FAR  = -150.0  # |x| > 0.5 — longe das bandeiras


def _shape_reward(reward, obs, terminated):
    """
    Shaping integrado no agente (sem wrapper externo).
    +0.2 por passo: reduz custo efectivo do combustível.
    Terminal com ambas as pernas: bónus/penalização por posição x.
    """
    reward += _STEP_BONUS
    if terminated:
        x_pos     = float(obs[0])
        left_leg  = float(obs[6])
        right_leg = float(obs[7])
        if left_leg > 0.5 and right_leg > 0.5:
            if abs(x_pos) <= 0.2:
                reward += _BONUS_CENTER
            elif abs(x_pos) <= 0.5:
                reward += _BONUS_NEAR
            else:
                reward += _PENALTY_FAR
    return reward


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
        q_values, _ = self.get_q(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(q_values)

    def train_episode(self, obs):
        action             = self.choose_action(obs)
        q_values, features = self.get_q(obs)
        e = np.zeros((self.num_actions, self.num_features))
        episode_reward = 0.0
        done = False
        while not done:
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            reward = _shape_reward(reward, next_obs, terminated)
            episode_reward += reward
            next_action               = self.choose_action(next_obs)
            next_q_values, next_feats = self.get_q(next_obs)
            if done:
                delta = reward - q_values[action]
            else:
                delta = reward + self.gamma * next_q_values[next_action] - q_values[action]
            delta = float(np.clip(delta, -50.0, 50.0))
            e        *= self.gamma * self.lam
            e[action]  = features
            self.weights += self.alpha_vec * delta * e
            np.clip(self.weights, -200.0, 200.0, out=self.weights)
            self.training_error.append(delta)
            obs, action, q_values, features = next_obs, next_action, next_q_values, next_feats
        return episode_reward

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "order": self.basis.order,
                         "low": self.low, "high": self.high}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        if "low" in data:
            self.low  = data["low"]
            self.high = data["high"]
        self.epsilon = 0.0