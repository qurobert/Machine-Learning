import pandas as pd
import numpy as np
from ML02.ex09.data_spliter import data_spliter
from ML02.ex07.polynomial_model import add_polynomial_features
from ML02.ex05.mylinearregression import MyLinearRegression as MyLR
import pickle
from matplotlib import pyplot as plt


def transform_features(X, power):
    transformed_features = []
    for i in range(X.shape[1]):  # Iterate over each feature column
        feature_column = X[:, i].reshape(-1, 1)  # Reshape the column to be a vector
        transformed_feature = add_polynomial_features(feature_column, power)
        transformed_features.append(transformed_feature)
    # Concatenate all transformed features horizontally
    return np.hstack(transformed_features)


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Load dataset
data = pd.read_csv('space_avocado.csv')
X = data[['weight', 'prod_distance', 'time_delivery']].values
Y = data['target'].values.reshape(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = data_spliter(X, Y, 0.8)

# Normalize features
X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)

# Load best model parameters
with open('models.pickle', 'rb') as f:
    model_params = pickle.load(f)

degree = 4  # Best degree
X_train_poly = transform_features(X_train_normalized, degree)
X_test_poly = transform_features(X_test_normalized, degree)
best_model = MyLR(model_params[f'model_degree_{degree}'])
best_model.fit_(X_train_poly, y_train)

# Evaluate and plot
y_pred = best_model.predict_(X_test_poly)
mse = best_model.loss_(y_test, y_pred)
print(f"MSE for degree {degree}: {mse}")

# Plot true vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True vs Predicted Prices')
plt.show()
