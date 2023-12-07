import numpy as np


def add_polynomial_features(x, power):
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    if x.size == 0 or x.shape[1] != 1:
        print(x.shape)
        return None
    # Initialize an empty array for the polynomial features
    poly_features = np.empty((x.shape[0], 0))
    # Iterate through powers from 1 to 'power' and concatenate the results
    for p in range(1, power + 1):
        poly_features = np.concatenate((poly_features, np.power(x, p).reshape(-1, 1)), axis=1)

    return poly_features

