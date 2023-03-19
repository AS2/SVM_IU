# *.py script with different some stock kernels

import numpy as np


def LinearKernel(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(x1, x2)
