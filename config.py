ENV_NAME = "LunarLander-v3"

N_EPISODES = 10000
MAX_STEPS  = 1000

LEARNING_RATE = 0.0005
GAMMA = 0.99
LAM   = 0.6
ORDER = 2

START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES * 0.75)
FINAL_EPSILON = 0.01

MODEL_PATH = "sarsa_lambda_lunar_lander.pkl"
PLOT_PATH  = "rewards_plot_lunar_lander.png"