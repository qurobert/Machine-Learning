import numpy as np


def data_spliter(x, y, proportion):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if x.size == 0 or y.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    if not isinstance(proportion, float):
        return None
    if proportion < 0 or proportion > 1:
        return None

    # Combine x and y to shuffle them together
    combined = np.hstack((x, y))
    np.random.shuffle(combined)

    # Split the combined array back into x and y components
    split_idx = int(combined.shape[0] * proportion)
    x_train = combined[:split_idx, :-y.shape[1]]
    x_test = combined[split_idx:, :-y.shape[1]]
    y_train = combined[:split_idx, -y.shape[1]:]
    y_test = combined[split_idx:, -y.shape[1]:]

    return x_train, x_test, y_train, y_test
