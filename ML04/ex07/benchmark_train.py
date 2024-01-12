import pandas as pd
import numpy as np
import pickle
from ML04.ex06.ridge import MyRidge  # Replace with your actual file name
from ML04.ex00.polynomial_model_extended import add_polynomial_features  # Replace with your actual file name
import matplotlib.pyplot as plt


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def split_data(X, y, train_size, cv_size, test_size):
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(X))

    train_indices = indices[:int(len(X) * train_size)]
    cv_indices = indices[int(len(X) * train_size):int(len(X) * (train_size + cv_size))]
    test_indices = indices[int(len(X) * (train_size + cv_size)):]

    return X[train_indices], X[cv_indices], X[test_indices], y[train_indices], y[cv_indices], y[test_indices]


# Load dataset
data = pd.read_csv('space_avocado.csv')
X = data[['weight', 'prod_distance', 'time_delivery']].values
y = data['target'].values.reshape(-1, 1)

# Split the dataset
X_train, X_cv, X_test, y_train, y_cv, y_test = split_data(X, y, 0.8, 0.1, 0.1)

# Normalize features
X_train_normalized = normalize(X_train)
X_cv_normalized = normalize(X_cv)
X_test_normalized = normalize(X_test)

# Training models
lambdas = np.arange(0, 1.1, 0.2)
degrees = range(1, 5)
models = {}

for degree in degrees:
    X_train_poly = add_polynomial_features(X_train_normalized, degree)
    X_cv_poly = add_polynomial_features(X_cv_normalized, degree)
    for lmbd in lambdas:
        model = MyRidge(np.ones((X_train_poly.shape[1] + 1, 1)), alpha=1e-2, max_iter=500000, lambda_=lmbd)
        model.fit_(X_train_poly, y_train)
        mse_cv = model.loss_(y_cv, model.predict_(X_cv_poly))
        models[(degree, lmbd)] = (model, mse_cv)
        print(f"Degree {degree}, Lambda {lmbd}, MSE {mse_cv}")

# Save the models to a pickle file
with open('models.pickle', 'wb') as f:
    pickle.dump(models, f)

# Find the best model based on MSE
best_degree, best_lambda, best_mse = None, None, float('inf')
for (degree, lmbd), (model, mse_cv) in models.items():
    if mse_cv < best_mse:
        best_mse = mse_cv
        best_degree, best_lambda = degree, lmbd

print(f"Best Model: Degree {best_degree}, Lambda {best_lambda}, MSE {best_mse}")

# Plotting MSE for each model
for degree in degrees:
    mses = [models[(degree, lmbd)][1] for lmbd in lambdas]
    plt.plot(lambdas, mses, label=f'Degree {degree}')

plt.xlabel('Lambda')
plt.ylabel('MSE on Cross-Validation Set')
plt.title('Model Performance')
plt.legend()
plt.show()

# Plotting predictions of the best model
best_model = models[(best_degree, best_lambda)][0]
X_test_poly_best = add_polynomial_features(X_test_normalized, best_degree)
y_pred_best = best_model.predict_(X_test_poly_best)

plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Predictions of Best Model (Degree {best_degree}, Lambda {best_lambda})')
plt.show()
