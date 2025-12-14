# utils_part2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# on réutilise ta fonction Partie I (AUC + ROC déjà dedans)
from utils import run_classifiers_train_test


# ============================================================
# LECTURE DATASET (robuste : virgule OU tab/espaces)
# ============================================================

def _read_credit_data(filename="credit.data"):
    """
    - Import via pandas.read_csv
    - Séparateur : virgule OU tab/espaces (robuste)
    - On lit tout en str (pour gérer '?')
    Retourne : df, data(np.array), X(np.array), y_txt(np.array '+'/'-')
    """
    # essai 1 : CSV virgule
    df = pd.read_csv(filename, header=None, sep=",", dtype=str)
    if df.shape[1] <= 1:
        # essai 2 : tab/espaces
        df = pd.read_csv(filename, header=None, sep=r"\s+", engine="python", dtype=str)

    data = df.values
    X = data[:, :-1].astype(object)   # 15 variables
    y_txt = data[:, -1].astype(str)   # target '+' / '-'
    return df, data, X, y_txt


def _binarize_target(y_txt):
    """ + -> 1 ; - -> 0 """
    return (y_txt == "+").astype(int)


def _plot_hist_classes(y_bin, title="Histogramme des classes"):
    n_pos = int((y_bin == 1).sum())
    n_neg = int((y_bin == 0).sum())
    plt.figure(figsize=(4, 3))
    plt.bar(["-", "+"], [n_neg, n_pos])
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()


def part2_run_numeric_no_missing(filename="credit.data", scaler="standard", show_raw=True):
    """
    Respect strict énoncé II.1 (version "style prof") + option show_raw
    - show_raw=True  : affiche brutes + normalisées
    - show_raw=False : affiche uniquement la normalisation (évite doublon quand tu testes 2 scalers)
    """

    print("\n" + "="*72)
    print("II.1) VARIABLES CONTINUES (NUMÉRIQUES) - SANS DONNÉES MANQUANTES")
    print("="*72)

    print("\n• Chargement des données et préparation :")
    df, data, X, y_txt = _read_credit_data(filename)
    print("  - Import pandas.read_csv ✅")
    print("  - Conversion DataFrame -> numpy (values) ✅")
    print("  - Séparation X (features) / y (target) ✅")
    print("  - Taille dataset brut (shape) :", df.shape)

    col_num = [1, 2, 7, 10, 13, 14]

    print("\n• Création sous-ensemble des variables numériques :")
    X_num = np.copy(X[:, col_num]).astype(object)
    X_num[X_num == "?"] = np.nan
    X_num = X_num.astype(float)
    print("  - Remplacement '?' -> NaN ✅")
    print("  - Typage float (astype) ✅")

    print("\n• Suppression des individus contenant des NaN :")
    mask = ~np.isnan(X_num).any(axis=1)
    X_num_clean = X_num[mask]
    y_clean = _binarize_target(y_txt)[mask]
    print("  - Shape après nettoyage :", X_num_clean.shape, "✅")

    print("\n• Analyse des propriétés :")
    n_pos = int((y_clean == 1).sum())
    n_neg = int((y_clean == 0).sum())
    print("  - Nombre d'exemples positifs (+) :", n_pos)
    print("  - Nombre d'exemples négatifs  (-) :", n_neg)
    print("  - Binarisation target (+->1, -->0) ✅")
    _plot_hist_classes(y_clean, title="Répartition des classes (+/-) - données numériques clean")

    print("\n• Split train/test 50/50 (stratifié) :")
    X_train, X_test, y_train, y_test = train_test_split(
        X_num_clean, y_clean, test_size=0.5, random_state=1, stratify=y_clean
    )
    print("  - Train :", X_train.shape, "| Test :", X_test.shape)

    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000),
    }

    if show_raw:
        print("\n• Comparaison des algorithmes (données numériques brutes) :")
        print("  - Critères affichés : Accuracy, Précision, Rappel, AUC + ROC ✅")
        run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test)

    print("\n• Normalisation des variables continues :")
    if scaler.lower() == "minmax":
        sc = MinMaxScaler()
        tag = "MinMaxScaler"
    else:
        sc = StandardScaler()
        tag = "StandardScaler"
    print(f"  - Normalisation utilisée : {tag} ✅")

    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    print("\n• Comparaison des algorithmes (données numériques normalisées) :")
    run_classifiers_train_test(clfs, X_train_sc, y_train, X_test_sc, y_test)




# ============================================================
# II.2) BASE COMPLÈTE (NUM + CAT) AVEC DONNÉES MANQUANTES
# Imputation mean (num) + most_frequent (cat)
# Encodage cat en int via np.unique(..., return_inverse=True)
# OneHotEncoder, puis concat num_norm + cat_onehot
# ============================================================

def part2_run_full_exact_statement(filename="credit.data", use_minmax=False):
    """
    Respect strict énoncé II.2 :
    - X_cat = np.copy(X[:, col_cat])
    - np.unique(... return_inverse=True) pour convertir cat en entiers
    - imputation cat most_frequent
    - num : '?' -> NaN -> float -> imputation mean
    - OneHotEncoder sur cat (9 vars)
    - normalisation sur num
    - concat final puis run_classifiers
    """

    print("\n" + "="*72)
    print("II.2) BASE COMPLÈTE (NUM + CAT) - DONNÉES MANQUANTES + ONE-HOT")
    print("="*72)

    print("\n• Chargement et séparation X / y :")
    df, data, X, y_txt = _read_credit_data(filename)
    y = _binarize_target(y_txt)
    print("  - Taille dataset brut (shape) :", df.shape, "✅")
    print("  - Positifs (+) :", int((y == 1).sum()), "| Négatifs (-) :", int((y == 0).sum()), "✅")

    # indices UCI
    col_num = [1, 2, 7, 10, 13, 14]
    col_cat = [0, 3, 4, 5, 6, 8, 9, 11, 12]

    # -------- NUM --------
    print("\n• Imputation des variables numériques (mean) :")
    X_num = np.copy(X[:, col_num]).astype(object)
    X_num[X_num == "?"] = np.nan
    X_num = X_num.astype(float)
    imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_num = imp_num.fit_transform(X_num)
    print("  - '?' -> NaN -> float -> imputation mean ✅")

    print("\n• Normalisation des variables numériques :")
    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    print("  - Normalisation :", "MinMaxScaler ✅" if use_minmax else "StandardScaler ✅")
    X_num = scaler.fit_transform(X_num)

    # -------- CAT --------
    print("\n• Traitement des variables catégorielles :")
    X_cat = np.copy(X[:, col_cat]).astype(object)

    # IMPORTANT : pour coller à l'esprit "missing_values=0", on force '?' -> '0'
    X_cat[X_cat == "?"] = "0"
    print("  - Valeurs manquantes '?' codées en 0 ✅")

    # Conversion en entiers via np.unique(... return_inverse=True)
    for col_id in range(len(col_cat)):
        unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
        X_cat[:, col_id] = val_idx
    X_cat = X_cat.astype(float)  # imputer attend float
    print("  - Encodage des catégories en entiers (np.unique + return_inverse) ✅")

    print("\n• Imputation des variables catégorielles (most_frequent) :")
    imp_cat = SimpleImputer(missing_values=0, strategy="most_frequent")
    X_cat = imp_cat.fit_transform(X_cat)
    print("  - Imputation most_frequent avec missing_values=0 ✅")

    print("\n• OneHotEncoder sur les 9 variables catégorielles :")
    try:
        ohe = OneHotEncoder(sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(sparse=False)
    X_cat_bin = ohe.fit_transform(X_cat)
    print("  - Shape X_cat_bin :", X_cat_bin.shape, "✅")

    print("\n• Construction du dataset final (concat num_norm + cat_onehot) :")
    X_final = np.hstack([X_num, X_cat_bin])
    print("  - Shape X_final :", X_final.shape, "✅")

    print("\n• Split train/test 50/50 (stratifié) :")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.5, random_state=1, stratify=y
    )
    print("  - Train :", X_train.shape, "| Test :", X_test.shape)

    print("\n• Exécution de run_classifiers (sur base complète préparée) :")
    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000),
        "RF_200": RandomForestClassifier(n_estimators=200, random_state=1),
        "AdaBoost_200": AdaBoostClassifier(n_estimators=200, random_state=1),
    }
    run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test)
