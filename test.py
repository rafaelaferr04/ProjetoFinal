import gymnasium as gym

from agent import SarsaLambdaAgent
from config import ENV_NAME, MAX_STEPS, ORDER, MODEL_PATH


def test(render=True, num_episodes=10):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)

    agent = SarsaLambdaAgent(env, order=ORDER)
    agent.load(MODEL_PATH)

    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episodic_reward = 0.0
        done  = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episodic_reward += reward
            steps += 1

        rewards.append(episodic_reward)
        print(f"Teste {episode + 1:2d}: reward = {episodic_reward:8.2f}, steps = {steps}")

    env.close()
    print(f"\nMédia: {sum(rewards) / len(rewards):.2f}")
    return rewards