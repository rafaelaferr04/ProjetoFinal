import numpy as np
import gymnasium as gym
from tqdm import tqdm

from agent import SarsaLambdaAgent
from config import ENV_NAME, N_EPISODES, ORDER, MODEL_PATH


def train():
    env = gym.make(ENV_NAME)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = SarsaLambdaAgent(env, order=ORDER)
    print(f"SARSA(λ) traces substitutivas — {agent.num_features} features por ação")
    print(f"α={agent.alpha}, γ={agent.gamma}, λ={agent.lam}, episódios={N_EPISODES}")
    for episode in tqdm(range(N_EPISODES)):
        obs, _ = env.reset()
        agent.train_episode(obs)
        agent.decay_epsilon()
        if (episode + 1) % 500 == 0:
            recent = list(env.return_queue)[-100:]
            avg = np.mean(recent) if recent else 0.0
            tqdm.write(f"  ep {episode+1:5d} | média 100: {avg:8.2f} | ε={agent.epsilon:.3f} | |w|={np.abs(agent.weights).max():.1f}")
    agent.save(MODEL_PATH)
    env.close()
    return env, agent