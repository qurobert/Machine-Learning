import numpy as np


def sigmoid_(x):
    if x.size == 0:
        return None
    return 1 / (1 + np.exp(-x))
