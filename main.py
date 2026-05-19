import sys

from train import train
from test import test
from utils import plot_rewards
from config import PLOT_PATH


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usa:")
        print("  python main.py train")
        print("  python main.py test")

    elif sys.argv[1] == "train":
        env, agent = train()
        plot_rewards(env, agent, PLOT_PATH)

    elif sys.argv[1] == "test":
        test(render=True, num_episodes=10)

    else:
        print("Argumento desconhecido.")