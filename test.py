import os

import gymnasium as gym

from agent import SarsaLambdaAgent
from config import (
    ENV_NAME, MAX_STEPS, ORDER, MODEL_PATH, BEST_MODEL_PATH, FLAG_HALF_WIDTH,
)


def _landed_in_centre(obs):
    return obs[6] > 0.5 and obs[7] > 0.5 and abs(obs[0]) <= FLAG_HALF_WIDTH


def test(render=True, num_episodes=10):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)

    agent = SarsaLambdaAgent(env, order=ORDER)
    # Preferir o melhor modelo guardado durante o treino; cair para o último se
    # não existir.
    path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    print(f"A carregar modelo: {path}")
    agent.load(path)

    rewards = []
    centre_landings = 0
    any_landings    = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episodic_reward = 0.0
        done  = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episodic_reward += reward
            steps += 1

        rewards.append(episodic_reward)
        in_centre = _landed_in_centre(obs)
        landed    = obs[6] > 0.5 and obs[7] > 0.5
        if landed:
            any_landings += 1
        if in_centre:
            centre_landings += 1

        tag = "CENTRO" if in_centre else ("aterrou" if landed else "falhou")
        print(
            f"Teste {episode + 1:2d}: reward={episodic_reward:8.2f}"
            f" | steps={steps:4d} | x_final={obs[0]:+.2f} | {tag}"
        )

    env.close()
    print(
        f"\nMédia recompensa: {sum(rewards) / len(rewards):.2f}"
        f" | Aterragens entre bandeiras: {centre_landings}/{num_episodes}"
        f" | Aterragens totais: {any_landings}/{num_episodes}"
    )
    return rewards
