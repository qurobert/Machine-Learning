from ML03.ex06.my_logistic_regression import MyLogisticRegression as MyLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multi_log():
    # Load dataset
    x_csv = pd.read_csv('solar_system_census.csv')
    y_csv = pd.read_csv('solar_system_census_planets.csv')
    x_data = x_csv[['weight', 'height', 'bone_density']].values
    y_data = y_csv['Origin'].values.reshape(-1, 1)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = MyLR.data_spliter(x_data, y_data, 0.8)

    models = []
    predictions = []

    # Train a model for each planet
    for planet in range(4):
        y_train_labeled = np.where(y_train == planet, 1, 0)
        y_test_labeled = np.where(y_test == planet, 1, 0)

        model = MyLR([1.0, 1.0, 1.0, 1.0], alpha=0.001, max_iter=500000)
        model.fit_(x_train, y_train_labeled)
        models.append(model)
        predictions.append(model.predict_(x_test))

    # Combine predictions and choose the class with the highest probability
    predictions = np.hstack(predictions)
    print(predictions)
    final_predictions = np.argmax(predictions, axis=1).reshape(-1, 1)
    print(final_predictions)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_test, final_predictions)
    print("Overall Accuracy:", accuracy)

    # Plotting each combination of features
    feature_names = ['Weight', 'Height', 'Bone Density']
    plot_features(x_test, final_predictions, 0, 1, feature_names)  # Weight vs Height
    plot_features(x_test, final_predictions, 0, 2, feature_names)  # Weight vs Bone Density
    plot_features(x_test, final_predictions, 1, 2, feature_names)  # Height vs Bone Density


def plot_features(x, y_class, feature1, feature2, feature_names):
    # Define colors and labels for each planet
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['Planet 0', 'Planet 1', 'Planet 2', 'Planet 3']

    plt.figure(figsize=(8, 6))

    # Plot each class with a unique color and label
    for i in range(4):
        mask = y_class.squeeze() == i
        plt.scatter(x[mask, feature1], x[mask, feature2], color=colors[i], label=labels[i], edgecolor='k')

    plt.xlabel(feature_names[feature1])
    plt.ylabel(feature_names[feature2])
    plt.title('Logistic Regression - One vs All')
    plt.legend()
    plt.show()


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


multi_log()
