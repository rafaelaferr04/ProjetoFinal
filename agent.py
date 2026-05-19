import numpy as np
import pickle

from fourier_basis import FourierBasis
from config import LEARNING_RATE, GAMMA, LAM, ORDER, START_EPSILON, FINAL_EPSILON, EPSILON_DECAY

# Shaping terminal: só modifica o reward quando o episódio termina com aterragem suave.
# NÃO existe bónus por passo — isso ensinava o agente a voar em vez de aterrar.
_BONUS_CENTER = 300.0   # aterragem entre bandeiras (|x| ≤ 0.15): +300 extra
_PENALTY_FAR  = 200.0   # aterragem fora das bandeiras (|x| > 0.15): -200 extra


def _shape_reward(reward, obs, terminated):
    """
    Shaping APENAS terminal: não altera rewards de passos intermédios.

    O ambiente já tem shaped rewards por passo (proximidade ao pad,
    velocidade, ângulo). Adicionar bónus por passo distorce o incentivo
    e ensina o agente a pairar em vez de aterrar.

    Só na aterragem suave (ambas as pernas): bónus se entre bandeiras,
    penalização se fora. Crash não é alterado (-100 do ambiente chega).
    """
    if terminated:
        x_pos     = float(obs[0])
        left_leg  = float(obs[6])
        right_leg = float(obs[7])
        if left_leg > 0.5 and right_leg > 0.5:   # aterragem suave
            if abs(x_pos) <= 0.15:
                reward += _BONUS_CENTER            # entre as bandeiras
            else:
                reward -= _PENALTY_FAR             # fora das bandeiras
    return reward


class SarsaLambdaAgent:
    """
    SARSA(λ) com base de Fourier e traces SUBSTITUTIVAS para LunarLander-v3.

    Baseado em sarsa_lambda_fourier_mountain_car.ipynb e taxi_sarsa_lambda.py.

    Traces SUBSTITUTIVAS (e[a] = features, não +=):
      - Limitadas a [-1,1]: nunca divergem (as acumulativas chegaram a 10^306)
    SARSA on-policy (usa Q(s',a') real):
      - Evita a "tríade mortal" que fazia o Q-Learning degradar após ep 2500
    Shaping integrado (_shape_reward):
      - Só terminal: não distorce o incentivo de aterragem
    """

    def __init__(self, env, order=ORDER):
        self.env         = env
        self.state_dim   = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        # Bounds do env para normalização — igual aos notebooks
        low  = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        self.low  = np.where(np.isfinite(low),  low,  -1.0)
        self.high = np.where(np.isfinite(high), high,  1.0)

        # Base de Fourier — igual ao notebook
        self.basis        = FourierBasis(self.state_dim, order)
        self.num_features = self.basis.num_features

        self.alpha   = LEARNING_RATE
        self.gamma   = GAMMA
        self.lam     = LAM

        self.epsilon       = START_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.final_epsilon = FINAL_EPSILON

        # Learning rate por feature: α / max(1, ‖c_i‖)
        norms = np.maximum(1.0, np.linalg.norm(self.basis.c, axis=1))
        self.alpha_vec = self.alpha / norms  # shape: (num_features,)

        self.weights = np.zeros((self.num_actions, self.num_features))
        self.training_error = []

    def normalize_state(self, state):
        return np.clip((state - self.low) / (self.high - self.low), 0.0, 1.0)

    def get_q(self, state):
        features = self.basis.get_features(self.normalize_state(state))
        return np.dot(self.weights, features), features

    def choose_action(self, state):
        """Epsilon-greedy — igual aos notebooks."""
        q_values, _ = self.get_q(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(q_values)

    def train_episode(self, obs):
        """
        SARSA(λ) com traces SUBSTITUTIVAS e shaping terminal integrado.

          reward   = env_reward + _shape_reward(...)   ← só no terminal
          delta    = r + γ·Q(s',a')·(não done) - Q(s,a)
          e        = γλ·e
          e[a]     = features(s)                       ← SUBSTITUI (não +=)
          w       += α_vec · clip(delta,-50,50) · e
        """
        action             = self.choose_action(obs)
        q_values, features = self.get_q(obs)
        e = np.zeros((self.num_actions, self.num_features))
        episode_reward = 0.0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Shaping apenas no passo terminal
            reward = _shape_reward(reward, next_obs, terminated)
            episode_reward += reward

            next_action               = self.choose_action(next_obs)
            next_q_values, next_feats = self.get_q(next_obs)

            # Erro TD on-policy (SARSA)
            if done:
                delta = reward - q_values[action]
            else:
                delta = (
                    reward
                    + self.gamma * next_q_values[next_action]
                    - q_values[action]
                )

            # Clipping: shaping terminal pode gerar deltas de ±400
            delta = float(np.clip(delta, -50.0, 50.0))

            # Traces SUBSTITUTIVAS: limitadas, não divergem
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
        self.epsilon = 0.0  # greedy durante teste