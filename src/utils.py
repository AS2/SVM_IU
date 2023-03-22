import numpy as np


def absmin(x):
    i = np.argmin(x)
    sign = 1 if x[i] >= 0 else 0
    newArr = [abs(x_i) * sign for x_i in x]
    return np.min(newArr)
