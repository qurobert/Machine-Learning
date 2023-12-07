import numpy as np
from ML02.ex05.mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import matplotlib.pyplot as plt


class MultivariateLinearModel:
    def __init__(self):
        self.data = pd.read_csv("./spacecraft_data.csv")

    def plot(self, X, Y, Y_pred, feature_name):
        plt.scatter(X, Y, color='blue', label='Sell Price')
        plt.plot(X, Y_pred, color='red', label='Predicted sell price')
        plt.xlabel(f'X: {feature_name}')
        plt.ylabel('Y: Price')
        plt.title(f'Actual vs Predicted Prices: {feature_name}')
        plt.legend()
        plt.show()

    def plot_multi(self, X, Y, Y_pred, feature_name):
        plt.scatter(X, Y, color='blue', label='Sell Price', s=50)
        plt.scatter(X, Y_pred, color='red', label='Predicted sell price', s=20)
        plt.xlabel(f'X: {feature_name}')
        plt.ylabel('Y: Price')
        plt.title(f'Actual vs Predicted Prices: {feature_name}')
        plt.legend()
        plt.show()

    def train_and_plot_feature_univariate(self, feature_name):
        X = np.array(self.data[feature_name]).reshape(-1, 1)
        Y = np.array(self.data["Sell_price"]).reshape(-1, 1)
        linear_model = MyLR(np.array([[1000.0], [-1.0]]), 2.5e-5, 1000000)
        linear_model.fit_(X, Y)  # Train the model
        Y_pred = linear_model.predict_(X)  # Predict the model

        # Outputting results
        print(f"Final Thetas for {feature_name}: {linear_model.thetas}")
        print(f"MSE for {feature_name}: {linear_model.loss_(Y, Y_pred)}")

        # Plotting
        self.plot(X, Y, Y_pred, feature_name)

    def age_feature(self):
        self.train_and_plot_feature_univariate("Age")

    def thrust_feature(self):
        self.train_and_plot_feature_univariate("Thrust_power")

    def total_distance_feature(self):
        self.train_and_plot_feature_univariate("Terameters")

    def train_and_plot_feature_multivariate(self, feature_names):
        X = np.array(self.data[feature_names])
        Y = np.array(self.data["Sell_price"]).reshape(-1, 1)
        initial_thetas = np.array([[1000.0]] + [[-1.0]] * len(feature_names))
        linear_model = MyLR(initial_thetas, 2.5e-5, 1000000)
        linear_model.fit_(X, Y)  # Train the model
        Y_pred = linear_model.predict_(X)  # Predict the model

        # Outputting results
        print(f"Final Thetas for {feature_names}: {linear_model.thetas}")
        print(f"MSE for {feature_names}: {linear_model.loss_(Y, Y_pred)}")

        # # Plotting
        for feature in feature_names:
            self.plot_multi(self.data[feature], Y, Y_pred, feature)


model_age = MultivariateLinearModel()
model_age.age_feature()

model_feature = MultivariateLinearModel()
model_feature.thrust_feature()

model_total_distance = MultivariateLinearModel()
model_total_distance.total_distance_feature()

multi_model = MultivariateLinearModel()
multi_model.train_and_plot_feature_multivariate(["Age", "Thrust_power", "Terameters"])