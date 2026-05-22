import copy
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from agent import SarsaLambdaAgent
from config import (
    ENV_NAME, MOON_GRAVITY, N_EPISODES, ORDER, MODEL_PATH, BEST_MODEL_PATH,
    FLAG_HALF_WIDTH, EVAL_EVERY, EVAL_EPISODES,
)


def _landed_in_centre(obs):
    legs = obs[6] > 0.5 and obs[7] > 0.5
    return legs and abs(obs[0]) <= FLAG_HALF_WIDTH


def _evaluate(env, agent, num_episodes):
    """Avaliação determinística (ε=0 temporário, sem pilot)."""
    saved_eps   = agent.epsilon
    saved_pilot = agent.pilot_prob
    agent.epsilon    = 0.0
    agent.pilot_prob = 0.0

    centre  = 0
    landed  = 0
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.choose_action(obs, training=False)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
        rewards.append(total)
        if obs[6] > 0.5 and obs[7] > 0.5:
            landed += 1
        if _landed_in_centre(obs):
            centre += 1

    agent.epsilon    = saved_eps
    agent.pilot_prob = saved_pilot
    return centre, landed, float(np.mean(rewards))


def train():
    env      = gym.make(ENV_NAME, gravity=MOON_GRAVITY)
    env      = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    eval_env = gym.make(ENV_NAME, gravity=MOON_GRAVITY)

    agent = SarsaLambdaAgent(env, order=ORDER)

    print(
        f"SARSA(λ) + Pilot heurístico — Base de Fourier — "
        f"{agent.num_features} features por ação"
    )
    print(
        f"α={agent.alpha}, γ={agent.gamma}, λ={agent.lam}, "
        f"episódios={N_EPISODES}, gravidade={MOON_GRAVITY} (Lua)"
    )

    best_centre  = -1
    best_mean_r  = -1e9
    best_weights = None

    for episode in tqdm(range(N_EPISODES)):
        obs, _ = env.reset()
        agent.train_episode(obs)
        agent.decay_epsilon()

        if (episode + 1) % EVAL_EVERY == 0:
            centre, landed, mean_r = _evaluate(eval_env, agent, EVAL_EPISODES)
            recent = list(env.return_queue)[-100:]
            avg = float(np.mean(recent)) if recent else 0.0
            tqdm.write(
                f"  ep {episode + 1:5d} | treino avg100={avg:7.1f}"
                f" | eval centro={centre:2d}/{EVAL_EPISODES}"
                f" aterrou={landed:2d}/{EVAL_EPISODES}"
                f" eval_r={mean_r:7.1f}"
                f" | ε={agent.epsilon:.3f}"
                f" | |w|_max={np.abs(agent.weights).max():.2f}"
            )

            score = (centre, mean_r)
            if best_weights is None or score > (best_centre, best_mean_r):
                best_centre   = centre
                best_mean_r   = mean_r
                best_weights  = copy.deepcopy(agent.weights)
                agent.save(BEST_MODEL_PATH)
                tqdm.write(f"    ↳ novo melhor modelo guardado em {BEST_MODEL_PATH}")

    agent.save(MODEL_PATH)
    env.close()
    eval_env.close()
    print(
        f"\nMelhor modelo: centro={best_centre}/{EVAL_EPISODES} "
        f"(guardado em {BEST_MODEL_PATH})"
    )
    return env, agent
