import gymnasium as gym

from ppo_agent import PPOAgent
from config import *


def test(render=True, num_episodes=10):
    if render:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = PPOAgent(
        num_states=num_states,
        num_actions=num_actions,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        clip_epsilon=CLIP_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        hidden_size=HIDDEN_SIZE
    )

    agent.load(MODEL_PATH)

    rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episodic_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_greedy_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episodic_reward += reward
            state = next_state
            steps += 1

        rewards.append(episodic_reward)

        print(
            f"Teste {episode + 1}: "
            f"reward = {episodic_reward:.2f}, steps = {steps}"
        )

    env.close()

    return rewards