import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y_true.shape != y_hat.shape:
        return None

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    else:
        labels = np.array(labels)

    conf_matrix = np.zeros((len(labels), len(labels)))

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            conf_matrix[i, j] = np.sum((y_true == label_i) & (y_hat == label_j))

    if df_option:
        return pd.DataFrame(conf_matrix, index=labels, columns=labels)
    return conf_matrix
