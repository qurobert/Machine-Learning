import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ML04.ex00.polynomial_model_extended import add_polynomial_features


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def load_models(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load dataset
data = pd.read_csv('space_avocado.csv')
X = data[['weight', 'prod_distance', 'time_delivery']].values
y = data['target'].values.reshape(-1, 1)

# Normalize features
X_normalized = normalize(X)

# Load trained models
models = load_models('models.pickle')

# Find the best model based on MSE
best_degree, best_lambda, best_model = None, None, None
lowest_mse = float('inf')

for (degree, lmbd), (model, mse) in models.items():
    if mse < lowest_mse:
        lowest_mse = mse
        best_degree, best_lambda, best_model = degree, lmbd, model

print(f"Best Model: Degree {best_degree}, Lambda {best_lambda}, MSE {lowest_mse}")

# Plotting predictions of the best model
X_poly_best = add_polynomial_features(X_normalized, best_degree)
y_pred_best = best_model.predict_(X_poly_best)

plt.scatter(y, y_pred_best, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Predictions of Best Model (Degree {best_degree}, Lambda {best_lambda})')
plt.show()
