import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ML02.ex07.polynomial_model import add_polynomial_features
from ML02.ex05.mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("./are_blue_pills_magics.csv")
X = np.array(data['Micrograms']).reshape(-1, 1)
Y = np.array(data['Score']).reshape(-1, 1)

mse_scores = []
models = []
degrees = range(1, 7)

for degree in degrees:
    X_poly = add_polynomial_features(X, degree)
    if degree == 4:
        initial_theta = np.array([[-20], [160], [-80], [10], [-1]], dtype=np.float64).reshape(-1, 1)
    elif degree == 5:
        initial_theta = np.array([[1140], [-1850], [1110], [-305], [40], [-2]], dtype=np.float64).reshape(-1, 1)
    elif degree == 6:
        initial_theta = np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]], dtype=np.float64).reshape(-1, 1)
    else:
        initial_theta = np.ones((degree + 1, 1), dtype=np.float64)
    model = MyLR(initial_theta, alpha=1e-9, max_iter=10000000)
    model.fit_(X_poly, Y)
    models.append(model)

    Y_pred = model.predict_(X_poly)
    mse = model.loss_(Y, Y_pred)
    mse_scores.append(mse)

# Plotting MSE Scores
plt.bar(range(1, 7), mse_scores)
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('MSE Scores by Polynomial Degree')
plt.show()

# Plotting Polynomial Regression Models
plt.scatter(X, Y, color='blue', label='Actual Data')
continuous_x = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)
for degree, model in zip(range(1, 7), models):
    continuous_x_poly = add_polynomial_features(continuous_x, degree)
    Y_continuous_pred = model.predict_(continuous_x_poly)
    plt.plot(continuous_x, Y_continuous_pred, label=f'Degree {degree}')
plt.xlabel('Micrograms')
plt.ylabel('Score')
plt.title('Polynomial Regression Models')
plt.legend()
plt.show()
