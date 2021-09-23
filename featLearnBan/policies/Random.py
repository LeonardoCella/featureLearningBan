from .Policy import Policy

import numpy as np


class Random(Policy):
    def __init__(self, K):
        self._K = K

    def choice(self, j, t, X):  # t is the round, x is a K*d matrix
        return np.random.choice(range(self._K))
