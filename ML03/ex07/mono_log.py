from ML03.ex06.my_logistic_regression import MyLogisticRegression as MyLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_features(x, y_prob, feature1, feature2, feature_names):
    plt.scatter(x[:, feature1], x[:, feature2], c=y_prob.squeeze(), cmap='viridis', edgecolor='k')
    plt.colorbar(label='Probability of Being From Favorite Planet')
    plt.xlabel(feature_names[feature1])
    plt.ylabel(feature_names[feature2])
    plt.title('Logistic Regression Predictions')
    plt.show()


def calculate_accuracy(y_true, y_pred_prob, threshold=0.5):
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    correct_predictions = np.sum(y_pred_binary == y_true)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def mono_log(x):
    if not isinstance(x, int) or x < 0 or x > 3:
        return print('x must be 0, 1, 2 or 3')

    # Load dataset
    x_csv = pd.read_csv('solar_system_census.csv')
    y_csv = pd.read_csv('solar_system_census_planets.csv')
    x_data = x_csv[['weight', 'height', 'bone_density']].values
    y_data = y_csv['Origin'].values.reshape(-1, 1)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = MyLR.data_spliter(x_data, y_data, 0.8)

    # Make new numpy array from y_test array with x = 1 if citizen comes from your favorite planet, 0 otherwise
    y_train_labeled = np.where(y_train == x, 1, 0)
    y_test_labeled = np.where(y_test == x, 1, 0)

    # Train a logistic model to predict if a citizen comes from your favorite planet or not,
    # using your brand new label.
    my_lr = MyLR([1.0, 1.0, 1.0, 1.0], alpha=0.001, max_iter=500000)
    my_lr.fit_(x_train, y_train_labeled)
    y_pred = my_lr.predict_(x_test)

    # Calculate and display the fraction of correct predictions over the total number of
    # predictions based on the test set.
    accuracy = calculate_accuracy(y_test_labeled, y_pred)
    print("Accuracy:", accuracy)

    feature_names = ['Weight', 'Height', 'Bone Density']

    # Plotting each combination of features
    plot_features(x_test, y_pred, 0, 1, feature_names)  # Weight vs Height
    plot_features(x_test, y_pred, 0, 2, feature_names)  # Weight vs Bone Density
    plot_features(x_test, y_pred, 1, 2, feature_names)  # Height vs Bone Density
    return my_lr, y_pred, accuracy, x_test, y_test


mono_log(0)
