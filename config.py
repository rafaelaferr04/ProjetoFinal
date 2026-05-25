ENV_NAME = "LunarLander-v3"

# Gravidade da Lua
MOON_GRAVITY = -10.0

N_EPISODES = 3000
MAX_STEPS  = 1000

LEARNING_RATE = 0.0003
GAMMA = 0.99
LAM   = 0.4
ORDER = 2

START_EPSILON = 0.4
EPSILON_DECAY = START_EPSILON / (N_EPISODES * 0.7)
FINAL_EPSILON = 0.05

# Quando o agente decide explorar, com esta probabilidade usa o pilot
PILOT_GUIDED_PROB = 0.7

# Largura do landing pad
FLAG_HALF_WIDTH = 0.2

DELTA_CLIP   = 10.0
WEIGHTS_CLIP = 50.0
WEIGHT_DECAY = 1e-5

MODEL_PATH      = "sarsa_lambda_lunar_lander.pkl"
BEST_MODEL_PATH = "sarsa_lambda_lunar_lander_best.pkl"
PLOT_PATH       = "rewards_plot_lunar_lander.png"

# Avaliação determinística periódica durante o treino
EVAL_EVERY    = 250
EVAL_EPISODES = 30
