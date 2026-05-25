"""
Pilot heurístico (camada reativa).

Controlador simples que sabe aterrar o módulo lunar razoavelmente entre as
bandeiras sem aprendizagem. Inspirado no controlador de referência incluído no
ambiente Box2D do Gymnasium.

É usado:
  1. Para guiar a exploração do agente durante o treino (em vez de ações
     uniformemente aleatórias), acelerando muito a convergência.
  2. Como fallback opcional quando a política aprendida ainda é fraca.
"""

import numpy as np


def pilot_action(state):
    x, y, vx, vy, theta, omega, leg1, leg2 = state

    if leg1 > 0.5 and leg2 > 0.5:
        return 0

    angle_target = np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4)
    hover_target = 0.55 * abs(x)

    angle_todo = (angle_target - theta) * 0.5 - omega * 1.0
    hover_todo = (hover_target - y) * 0.5 - vy * 0.5

    if leg1 > 0.5 or leg2 > 0.5:
        angle_todo = 0.0
        hover_todo = -vy * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0
