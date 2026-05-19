import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(env, agent, path, rolling_length=50):
    """
    Gráfico de treino — 3 painéis, igual ao tutorial Blackjack:
      - Recompensas por episódio
      - Duração dos episódios
      - Erro TD de treino
    """
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Recompensas por episódio")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Duração dos episódios")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(),
            np.ones(rolling_length),
            mode="same",
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Erro TD (treino)")
    training_error_moving_average = (
        np.convolve(
            np.array(agent.training_error),
            np.ones(rolling_length),
            mode="same",
        )
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')   # fecha sem bloquear o terminal
    print(f"Gráfico guardado em {path}")