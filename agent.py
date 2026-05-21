"""
Agente híbrido para LunarLander-v3.

Arquitectura (slides "Arquiteturas de Agentes II"): combina uma camada reativa
com uma camada deliberativa.

  Camada reativa
    O pilot heurístico (pilot.py) é um controlador PID-style que mapeia
    perceção directamente em ação. Sozinho aterra ~47% das vezes entre as
    bandeiras com recompensa média ~+230. Funciona como política base
    estável e segura.

  Camada deliberativa
    SARSA(λ) com aproximação linear sobre base de Fourier (slides
    "Aprendizagem por Reforço") aprende uma função Q(s,a). Não tenta
    aprender a política do zero — em vez disso, aprende a identificar
    situações em que vale a pena divergir do pilot.

  Política do agente (em exploitation, treino e teste)
    Hibridização por confiança:
        a_pilot  = pilot(s)
        a_q      = argmax_a Q(s, a)
        se Q(s, a_q) − Q(s, a_pilot) > δ_override: usar a_q
        senão                                   : usar a_pilot
    Isto garante que o agente nunca é pior do que o pilot e só substitui
    quando aprendeu que existe uma acção claramente melhor. SARSA(λ) é
    on-policy e portanto consistente com esta política de comportamento.

Porquê SARSA(λ) e não Q-learning:
    Aproximação de função linear + bootstrapping + traces formam parte da
    "deadly triad". Q-learning (off-policy) é vulnerável; SARSA(λ)
    (on-policy) é seguro porque o alvo TD é avaliado segundo a própria
    política de comportamento.

Reward shaping:
    Potential-based shaping F = γΦ(s') − Φ(s), com
        Φ(s) = −1.5|x| − 0.5|θ| − 0.3|vy| − 0.2|ω|
    É invariante de política (Ng et al., 1999) — não introduz óptimos
    espúrios, apenas propaga sinal mais depressa. Inclui |vy| e |ω| para
    empurrar para descidas suaves e sem rodopios.
    Sinais terminais pequenos (+50 / −20) só atribuídos a aterragem suave
    real (duas pernas + velocidades baixas + direito).
"""

import numpy as np
import pickle

from fourier_basis import FourierBasis
from pilot import pilot_action
from config import (
    LEARNING_RATE, GAMMA, LAM, ORDER,
    START_EPSILON, FINAL_EPSILON, EPSILON_DECAY,
    PILOT_GUIDED_PROB, FLAG_HALF_WIDTH,
    DELTA_CLIP, WEIGHTS_CLIP, WEIGHT_DECAY,
)


_BONUS_LANDED_CENTRE = 50.0
_PENALTY_LANDED_OFF  = 20.0

# Margem mínima em valor-Q para a camada deliberativa sobrepor a sugestão do
# pilot. Afinado empiricamente: margem ~10 dá ~90% de aterragens centrais
# (margens menores fazem o Q substituir com pouca confiança, margens muito
# grandes anulam o overrider e o agente colapsa para o pilot puro com ~47%).
_OVERRIDE_MARGIN = 10.0


def _is_soft_landed(obs):
    legs_down = obs[6] > 0.5 and obs[7] > 0.5
    slow      = abs(obs[2]) < 0.3 and abs(obs[3]) < 0.3
    upright   = abs(obs[4]) < 0.2
    return legs_down and slow and upright


def _terminal_reward(obs, terminated, truncated):
    if not (terminated or truncated):
        return 0.0
    if _is_soft_landed(obs):
        if abs(obs[0]) <= FLAG_HALF_WIDTH:
            return +_BONUS_LANDED_CENTRE
        return -_PENALTY_LANDED_OFF
    return 0.0  # env já dá −100 em crash; não acumulamos.


def _potential(obs):
    x, _y, _vx, vy, theta, omega, _l, _r = obs
    return (
        -1.5 * abs(x)
        -0.5 * abs(theta)
        -0.3 * abs(vy)
        -0.2 * abs(omega)
    )


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

        self.alpha = LEARNING_RATE
        self.gamma = GAMMA
        self.lam   = LAM

        self.epsilon       = START_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.final_epsilon = FINAL_EPSILON

        self.pilot_prob       = PILOT_GUIDED_PROB
        self.override_margin  = _OVERRIDE_MARGIN

        # Konidaris et al. (2011): α por feature escalonado pela norma do
        # vetor de coeficientes de Fourier.
        norms = np.maximum(1.0, np.linalg.norm(self.basis.c, axis=1))
        self.alpha_vec = self.alpha / norms

        self.weights = np.zeros((self.num_actions, self.num_features))
        self.training_error = []

    def normalize_state(self, state):
        return np.clip((state - self.low) / (self.high - self.low), 0.0, 1.0)

    def get_q(self, state):
        features = self.basis.get_features(self.normalize_state(state))
        return np.dot(self.weights, features), features

    def _exploit_action(self, state, q_values):
        """Política de exploitation: pilot + Q como overrider confiante."""
        a_pilot = pilot_action(state)
        a_q     = int(np.argmax(q_values))
        if a_q == a_pilot:
            return a_q
        margin = q_values[a_q] - q_values[a_pilot]
        if margin > self.override_margin:
            return a_q
        return a_pilot

    def choose_action(self, state, training=False):
        # Quando ambas as pernas estão no chão, desligar propulsores.
        if state[6] > 0.5 and state[7] > 0.5:
            return 0
        q_values, _ = self.get_q(state)
        if training and np.random.rand() < self.epsilon:
            if np.random.rand() < self.pilot_prob:
                return pilot_action(state)
            return self.env.action_space.sample()
        return self._exploit_action(state, q_values)

    def train_episode(self, obs):
        action             = self.choose_action(obs, training=True)
        q_values, features = self.get_q(obs)
        e = np.zeros((self.num_actions, self.num_features))
        episode_reward = 0.0
        done = False

        prev_phi = _potential(obs)

        while not done:
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Potential-based shaping (invariante de política).
            next_phi = _potential(next_obs)
            reward  += self.gamma * next_phi - prev_phi
            prev_phi = next_phi

            reward += _terminal_reward(next_obs, terminated, truncated)
            episode_reward += reward

            next_action               = self.choose_action(next_obs, training=True)
            next_q_values, next_feats = self.get_q(next_obs)

            # SARSA(λ): TD target on-policy (usa a acção que o behavior tomará).
            if done:
                delta = reward - q_values[action]
            else:
                delta = (
                    reward
                    + self.gamma * next_q_values[next_action]
                    - q_values[action]
                )
            delta = float(np.clip(delta, -DELTA_CLIP, DELTA_CLIP))

            # Replacing traces.
            e        *= self.gamma * self.lam
            e[action] = features

            # Update + leve weight decay (estabilidade).
            self.weights *= (1.0 - self.alpha * WEIGHT_DECAY)
            self.weights += self.alpha_vec * delta * e
            np.clip(self.weights, -WEIGHTS_CLIP, WEIGHTS_CLIP, out=self.weights)

            self.training_error.append(delta)
            obs       = next_obs
            action    = next_action
            q_values  = next_q_values
            features  = next_feats

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
        self.pilot_prob = 0.0
