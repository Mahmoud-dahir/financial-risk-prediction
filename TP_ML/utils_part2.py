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

# Même routine d'évaluation que la Partie I (matrice de confusion, ROC/AUC, etc.)
from utils import run_classifiers_train_test


def _read_credit_data(filename="credit.data"):
    """
    Lecture du dataset UCI "credit approval".
    On charge tout en string pour gérer correctement les valeurs manquantes '?'.

    Le fichier peut être :
      - séparé par des virgules
      - ou séparé par des espaces / tabulations
    """
    df = pd.read_csv(filename, header=None, sep=",", dtype=str)
    if df.shape[1] <= 1:
        df = pd.read_csv(filename, header=None, sep=r"\s+", engine="python", dtype=str)

    data = df.values
    X = data[:, :-1].astype(object)   # 15 variables
    y_txt = data[:, -1].astype(str)   # '+' / '-'
    return df, X, y_txt


def _binarize_target(y_txt):
    """Cible binaire : '+' -> 1 ; '-' -> 0"""
    return (y_txt == "+").astype(int)


def _plot_hist_classes(y_bin, title="Répartition des classes"):
    """Petit histogramme de la répartition des classes."""
    n_pos = int((y_bin == 1).sum())
    n_neg = int((y_bin == 0).sum())
    plt.figure(figsize=(4, 3))
    plt.bar(["-", "+"], [n_neg, n_pos])
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()


def part2_run_numeric_no_missing(filename="credit.data", scaler="standard", show_raw=True):
    """
    II.1 : Variables numériques uniquement, sans valeurs manquantes.
    1) Extraction des colonnes numériques
    2) Suppression des individus incomplets
    3) Split 50/50 stratifié
    4) Comparaison des modèles (brut si show_raw=True)
    5) Comparaison après normalisation (StandardScaler ou MinMaxScaler)
    """
    print("\n" + "=" * 72)
    print("II.1) Variables numériques (sans valeurs manquantes)")
    print("=" * 72)

    df, X, y_txt = _read_credit_data(filename)
    print("Dataset shape (brut) :", df.shape)

    # Colonnes numériques (indices UCI)
    col_num = [1, 2, 7, 10, 13, 14]

    # Extraction + conversion float (avec gestion des '?')
    X_num = np.copy(X[:, col_num]).astype(object)
    X_num[X_num == "?"] = np.nan
    X_num = X_num.astype(float)

    # On enlève les lignes qui contiennent au moins un NaN
    mask = ~np.isnan(X_num).any(axis=1)
    X_num_clean = X_num[mask]
    y_clean = _binarize_target(y_txt)[mask]

    print("Shape après nettoyage :", X_num_clean.shape)
    _plot_hist_classes(y_clean, title="Répartition des classes (+/-) après nettoyage")

    # Split 50/50 stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_num_clean, y_clean, test_size=0.5, random_state=1, stratify=y_clean
    )
    print("Train :", X_train.shape, "| Test :", X_test.shape)

    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000),
    }

    if show_raw:
        print("\nComparaison sur données numériques brutes :")
        run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test)

    # Normalisation (fit sur train, transform sur test)
    if scaler.lower() == "minmax":
        sc = MinMaxScaler()
        tag = "MinMaxScaler"
    else:
        sc = StandardScaler()
        tag = "StandardScaler"

    print(f"\nComparaison après normalisation ({tag}) :")
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    run_classifiers_train_test(clfs, X_train_sc, y_train, X_test_sc, y_test)


def part2_run_full_exact_statement(filename="credit.data", use_minmax=False, verbose=True):
    """
    II.2 : Base complète (num + cat) avec données manquantes.

    Préparation :
      - Numériques : '?' -> NaN -> float -> imputation(mean) -> normalisation
      - Catégorielles : encodage en entiers (0 pour manquant) -> imputation(most_frequent) -> OneHotEncoder
      - Concat : [num_norm] + [cat_onehot]

    Ensuite :
      - Split 50/50 stratifié
      - Comparaison de plusieurs modèles
    """
    if verbose:
        print("\n" + "=" * 72)
        print("II.2) Base complète (num + cat) avec imputation + one-hot")
        print("=" * 72)

    df, X, y_txt = _read_credit_data(filename)
    y = _binarize_target(y_txt)

    if verbose:
        print("Dataset shape (brut) :", df.shape)
        print("Positifs :", int((y == 1).sum()), "| Négatifs :", int((y == 0).sum()))

    # Indices UCI
    col_num = [1, 2, 7, 10, 13, 14]
    col_cat = [0, 3, 4, 5, 6, 8, 9, 11, 12]

    # -----------------------
    # Partie numérique
    # -----------------------
    X_num = np.copy(X[:, col_num]).astype(object)
    X_num[X_num == "?"] = np.nan
    X_num = X_num.astype(float)

    imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_num = imp_num.fit_transform(X_num)

    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    X_num = scaler.fit_transform(X_num)

    # -----------------------
    # Partie catégorielle
    # -----------------------
    X_cat = np.copy(X[:, col_cat]).astype(object)
    X_cat_int = np.zeros(X_cat.shape, dtype=int)

    # Encodage colonne par colonne : 0 = manquant, 1..K = catégories
    for j in range(len(col_cat)):
        col = X_cat[:, j].astype(str)
        miss = (col == "?")

        col_no_miss = col[~miss]
        _, val_idx = np.unique(col_no_miss, return_inverse=True)

        X_cat_int[~miss, j] = val_idx + 1
        X_cat_int[miss, j] = 0

    imp_cat = SimpleImputer(missing_values=0, strategy="most_frequent")
    X_cat_imp = imp_cat.fit_transform(X_cat_int.astype(float))

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    X_cat_bin = ohe.fit_transform(X_cat_imp)

    # -----------------------
    # Concat + split + modèles
    # -----------------------
    X_final = np.hstack([X_num, X_cat_bin])

    if verbose:
        print("Shape num :", X_num.shape, "| Shape cat one-hot :", X_cat_bin.shape)
        print("Shape final :", X_final.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.5, random_state=1, stratify=y
    )

    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000),
        "RF_200": RandomForestClassifier(n_estimators=200, random_state=1),
        "AdaBoost_200": AdaBoostClassifier(n_estimators=200, random_state=1),
    }

    print("\nComparaison sur base complète préparée :")
    run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test)
