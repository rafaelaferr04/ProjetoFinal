ENV_NAME = "LunarLander-v3"

NUM_EPISODES = 1000
MAX_STEPS = 1000

GAMMA = 0.99
LEARNING_RATE = 0.0003
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4

HIDDEN_SIZE = 128

MODEL_PATH = "ppo_lunar_lander_model.pth"
PLOT_PATH = "rewards_plot_lunar_lander_ppo.png"