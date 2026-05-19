ENV_NAME = "LunarLander-v3"

N_EPISODES = 10000
MAX_STEPS  = 1000

# SARSA(λ) com base de Fourier e traces substitutivas
LEARNING_RATE = 0.0005  # ligeiramente mais baixo para estabilidade a longo prazo
GAMMA = 0.99            # padrão — era 0.995 só para compensar o step bonus (removido)
LAM   = 0.6             # traces moderadas: mais estáveis que 0.9, mais crédito que 0.5
ORDER = 2               # (ORDER+1)^8 = 6561 features

# Epsilon-greedy com decaimento linear ao longo de 75% dos episódios
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES * 0.75)
FINAL_EPSILON = 0.01    # muito greedy no fim: explora a política aprendida

MODEL_PATH = "sarsa_lambda_lunar_lander.pkl"
PLOT_PATH  = "rewards_plot_lunar_lander.png"