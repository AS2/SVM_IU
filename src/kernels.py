# *.py script with different some stock kernels

import numpy as np


def LinearKernel(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(x1, x2)


class GaussianKernel:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        pass

    def __similarity(self, x, l):
        # Вычисляем экспоненциальную функцию
        return np.exp(-sum((x - l) ** 2) / (2 * self.sigma))

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return self.__similarity(x1, x2)
