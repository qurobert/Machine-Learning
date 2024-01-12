import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y_true.shape != y_hat.shape:
        return None

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    else:
        labels = np.array(labels)

    conf_matrix = np.zeros((len(labels), len(labels)))

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            conf_matrix[i, j] = np.sum((y_true == label_i) & (y_hat == label_j))

    if df_option:
        return pd.DataFrame(conf_matrix, index=labels, columns=labels)
    return conf_matrix


def data_spliter(x, y, proportion):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if x.size == 0 or y.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    if not isinstance(proportion, float):
        return None
    if proportion < 0 or proportion > 1:
        return None

    # Combine x and y to shuffle them together
    combined = np.hstack((x, y))
    np.random.shuffle(combined)

    # Split the combined array back into x and y components
    split_idx = int(combined.shape[0] * proportion)
    x_train = combined[:split_idx, :-y.shape[1]]
    x_test = combined[split_idx:, :-y.shape[1]]
    y_train = combined[:split_idx, -y.shape[1]:]
    y_test = combined[split_idx:, -y.shape[1]:]

    return x_train, x_test, y_train, y_test


def normalize(x):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if x.size == 0:
        return None
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


# Chargez les modèles
with open('models.pickle', 'rb') as f:
    models = pickle.load(f)

# Charger et préparer les données
x_csv = pd.read_csv('solar_system_census.csv')
y_csv = pd.read_csv('solar_system_census_planets.csv')
x_data = x_csv[['weight', 'height', 'bone_density']].values
y_data = y_csv['Origin'].values.reshape(-1, 1)

# Split et normalisation des données
x_train, x_test, y_train, y_test = data_spliter(x_data, y_data, 0.8)
x_train = normalize(x_train)
x_test = normalize(x_test)

# Ajout de caractéristiques polynomiales
degree = 3
x_train_poly = add_polynomial_features(x_train, degree)
x_test_poly = add_polynomial_features(x_test, degree)

# Évaluation des modèles
lambdas = np.arange(0, 100.1, 50)

for planet in range(4):
    f1_scores = []
    y_true_planet = (y_test == planet).astype(int)

    for lmbd in lambdas:
        try:
            model, _ = models[(planet, lmbd)]
            y_pred_test = (model.predict_(x_test_poly) >= 0.5).astype(int)
            f1 = f1_score_(y_true_planet, y_pred_test)
            f1_scores.append(f1)
            print(f'Planet: {planet}, Lambda: {lmbd}, F1 Score: {f1}')
        except KeyError as e:
            print(f"Modèle non trouvé pour la clé : {e}")
            f1_scores.append(None)

    # Visualisation des performances pour chaque planète
    plt.figure()  # Crée une nouvelle figure pour chaque planète
    if f1_scores:  # Vérifie si la liste n'est pas vide
        plt.bar(lambdas, f1_scores, width=5, label=f'Planète {planet}')
        plt.xlabel('Lambda')
        plt.ylabel('F1 Score')
        plt.title(f'Performance des Modèles pour la Planète {planet}')
        plt.legend()
        plt.show()
    else:
        print(f"Aucune donnée de performance pour la planète {planet}")

    # Matrice de confusion
    cm = confusion_matrix_(y_true_planet, y_pred_test)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='0f', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.title(f'Matrice de confusion pour la planète {planet}')
    plt.show()