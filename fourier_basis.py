import numpy as np


class FourierBasis:
    """
    Base de Fourier para aproximação de funções.
    Copiado de: sarsa_lambda_fourier_mountain_car.ipynb
    """

    def __init__(self, state_dim, order):
        self.state_dim    = state_dim
        self.order        = order
        self.num_features = (order + 1) ** state_dim
        self.c = np.array(
            np.meshgrid(*[range(order + 1)] * state_dim)
        ).T.reshape(-1, state_dim)

    def get_features(self, state):
        # state deve estar normalizado para [0, 1]
        return np.cos(np.pi * np.dot(self.c, state))