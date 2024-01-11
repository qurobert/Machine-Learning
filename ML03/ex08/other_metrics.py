import numpy as np


def accuracy_score_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    return np.sum(y == y_hat) / len(y)


def precision_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
