import pandas as pd
import numpy as np
from ML02.ex09.data_spliter import data_spliter
from ML02.ex07.polynomial_model import add_polynomial_features
from ML02.ex05.mylinearregression import MyLinearRegression as MyLR
import pickle
from matplotlib import pyplot as plt


# Define transform_features function to apply add_polynomial_features to each column
def transform_features(X, power):
    transformed_features = []
    for i in range(X.shape[1]):
        feature_column = X[:, i].reshape(-1, 1)
        transformed_feature = add_polynomial_features(feature_column, power)
        transformed_features.append(transformed_feature)
    return np.hstack(transformed_features)


# Define normalize function
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Load dataset
data = pd.read_csv('space_avocado.csv')
X = data[['weight', 'prod_distance', 'time_delivery']].values
Y = data['target'].values.reshape(-1, 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = data_spliter(X, Y, 0.8)

# Initialize a dictionary to store models and their respective MSEs
model_params = {}
mse_scores = {}

# Train models for degrees 1 through 4
for degree in range(1, 5):
    print(f"Training degree {degree} model...")
    X_train_poly = transform_features(X_train_normalized, degree)
    X_test_poly = transform_features(X_test_normalized, degree)

    # Initialize MyLinearRegression with a vector of ones as thetas
    thetas = np.ones((X_train_poly.shape[1] + 1, 1))
    model = MyLR(thetas, alpha=1e-2, max_iter=1000000)
    model.fit_(X_train_poly, y_train)

    # Predict on the testing set
    y_pred = model.predict_(X_test_poly)

    # Compute and store the MSE
    mse = model.loss_(y_test, y_pred)
    mse_scores[degree] = mse
    print(f"MSE for degree {degree}: {mse}")

    # Save the model parameters
    model_params[f'model_degree_{degree}'] = model.thetas.tolist()  # Convert to list for compatibility

# Save all model parameters to a pickle file
with open('models.pickle', 'wb') as f:
    pickle.dump(model_params, f)

# Identify the best model (the one with the lowest MSE)
best_degree = min(mse_scores, key=mse_scores.get)
best_thetas = model_params[f'model_degree_{best_degree}']

# Plot the best model's true vs predicted prices
X_test_poly_best = transform_features(X_test_normalized, best_degree)
best_model = MyLR(np.array(best_thetas), alpha=1e-5, max_iter=50000)
y_pred_best = best_model.predict_(X_test_poly_best)

plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title(f'True vs Predicted Prices (Degree {best_degree})')
plt.show()

# Plot MSE scores for different models
plt.plot(list(mse_scores.keys()), list(mse_scores.values()), marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('MSE for Different Polynomial Degrees')
plt.show()