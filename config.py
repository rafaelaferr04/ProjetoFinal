ENV_NAME = "LunarLander-v3"

N_EPISODES = 8000
MAX_STEPS  = 1000

# SARSA(λ) com base de Fourier e traces substitutivas
LEARNING_RATE = 0.001
GAMMA = 0.995   # menos desconto por passo: valoriza mais a recompensa final
LAM   = 0.5
ORDER = 2

START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES * 0.80)
FINAL_EPSILON = 0.01

MODEL_PATH = "sarsa_lambda_lunar_lander.pkl"
PLOT_PATH  = "rewards_plot_lunar_lander.png"