ENV_NAME = "LunarLander-v3"

# Gravidade da Lua na convenção do env (constante física, não varia).
# O env aceita o parâmetro `gravity` em gym.make — passamos este valor
# explicitamente para deixar claro que estamos a simular a Lua e não a Terra.
MOON_GRAVITY = -10.0

N_EPISODES = 3000
MAX_STEPS  = 1000

# Aproximação linear + features densas (base de Fourier completa) → mantemos
# α e λ pequenos para evitar a "deadly triad" (FA + bootstrapping + traces).
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAM   = 0.4
ORDER = 2

# Exploração mais baixa: o pilot heurístico já fornece trajectórias úteis.
START_EPSILON = 0.4
EPSILON_DECAY = START_EPSILON / (N_EPISODES * 0.7)
FINAL_EPSILON = 0.05

# Quando o agente decide explorar, com esta probabilidade usa o pilot
# heurístico em vez de uma ação uniformemente aleatória.
PILOT_GUIDED_PROB = 0.7

# Largura do landing pad em LunarLander-v3 (pad em x ∈ [-0.2, 0.2]).
FLAG_HALF_WIDTH = 0.2

# Estabilidade numérica.
DELTA_CLIP   = 10.0
WEIGHTS_CLIP = 50.0
WEIGHT_DECAY = 1e-5

MODEL_PATH      = "sarsa_lambda_lunar_lander.pkl"
BEST_MODEL_PATH = "sarsa_lambda_lunar_lander_best.pkl"
PLOT_PATH       = "rewards_plot_lunar_lander.png"

# Avaliação determinística periódica durante o treino (grava o melhor modelo).
EVAL_EVERY    = 250
EVAL_EPISODES = 30
