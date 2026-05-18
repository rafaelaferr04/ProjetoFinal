import gymnasium as gym

from ppo_agent import PPOAgent
from config import *


def train():
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

    rewards_each_episode = []

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        episodic_reward = 0

        for step in range(MAX_STEPS):
            action, log_prob, value = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.states.append(state)
            agent.buffer.values.append(value)
            agent.buffer.actions.append(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            agent.buffer.log_probs.append(log_prob)

            episodic_reward += reward
            state = next_state

            if done:
                break

        agent.update()

        rewards_each_episode.append(episodic_reward)

        print(
            f"Episódio {episode + 1}/{NUM_EPISODES} "
            f"Reward: {episodic_reward:.2f}"
        )

    agent.save(MODEL_PATH)
    env.close()

    return rewards_each_episode