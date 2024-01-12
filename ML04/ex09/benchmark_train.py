import numpy as np
import pandas as pd
import pickle
from ML04.ex08.my_logistic_regression import MyLogisticRegression as MyLR
from ML04.ex00.polynomial_model_extended import add_polynomial_features


def precision_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def normalize(x):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if x.size == 0:
        return None
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


# Load dataset
x_csv = pd.read_csv('solar_system_census.csv')
y_csv = pd.read_csv('solar_system_census_planets.csv')
x_data = x_csv[['weight', 'height', 'bone_density']].values
y_data = y_csv['Origin'].values.reshape(-1, 1)

# Data Splitter Function
x_train, x_cv, x_test, y_train, y_cv, y_test = MyLR.data_spliter(x_data, y_data, 0.5, 0.25, 0.25)

# Normalize
x_train = normalize(x_train)
x_cv = normalize(x_cv)
x_test = normalize(x_test)

# Polynomial Features
degree = 3
x_train_poly = add_polynomial_features(x_train, degree)
x_cv_poly = add_polynomial_features(x_cv, degree)

# Training and Evaluation
models = {}
lambdas = np.arange(0, 100.1, 50)

for planet in range(4):
    y_train_binary = np.where(y_train == planet, 1, 0)
    y_cv_binary = np.where(y_cv == planet, 1, 0)

    for lmbd in lambdas:
        theta_init = np.random.randn(x_train_poly.shape[1] + 1, 1) * 0.01
        model = MyLR(theta_init, alpha=0.00001, max_iter=500000, lambda_=lmbd)
        model.fit_(x_train_poly, y_train_binary)
        y_pred_cv_probs = model.predict_(x_cv_poly)
        y_pred_cv_labels = (y_pred_cv_probs >= 0.5).astype(int)
        f1 = f1_score_(y_cv_binary, y_pred_cv_labels)
        print(f'Planet {planet}, Lambda {lmbd}, F1 Score {f1}, Thetas {model.theta}')
        models[(planet, lmbd)] = (model, f1)

# Save Models - make sure you have write permissions
with open('models.pickle', 'wb') as f:
    pickle.dump(models, f)
