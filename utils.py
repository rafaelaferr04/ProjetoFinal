import matplotlib.pyplot as plt


def plot_rewards(rewards, path):
    plt.plot(rewards)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title("Curva de aprendizagem - PPO Lunar Lander")
    plt.savefig(path)
    plt.show()