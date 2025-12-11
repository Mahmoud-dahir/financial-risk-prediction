import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import RFE   
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, make_scorer
)


warnings.filterwarnings("ignore")
np.set_printoptions(threshold=10000, suppress=True)

# ------------------------------------------------------------
# 0. CHARGEMENT ET SPLIT
# ------------------------------------------------------------

def load_and_split_data(filename, sep=';'):
    try:
        df = pd.read_csv(filename, sep=sep)
    except:
        df = pd.read_csv(filename, sep=',')
    data = df.values
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(float)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"Dataset : {df.shape}")
    print(f"Répartition : {n_pos} Positifs ({n_pos/len(y):.2%}) / {n_neg} Négatifs")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


# ------------------------------------------------------------
# 1. COMPARAISON DES CLASSIFIEURS (Q2)
# ------------------------------------------------------------

def evaluation(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = y_pred

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    best_criteria = prec
    score_final = (acc + best_criteria) / 2

    print(f"\n--- {model_name} ---")
    print(f"Matrice :\n{cm}")
    print(f"Accuracy : {acc:.3f} | Précision : {prec:.3f} | Rappel : {rec:.3f}")
    print(f"AUC : {auc:.3f}")
    print(f"👉 Score Final : {score_final:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

    return score_final


def run_classifiers_train_test(classifiers, X_train, y_train, X_test, y_test):
    results = {}
    best_score = 0
    best_model_obj = None
    best_model_name = ""

    plt.figure(figsize=(8, 6))
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        score = evaluation(model, X_test, y_test, name)
        results[name] = score
        if score > best_score:
            best_score = score
            best_model_obj = model
            best_model_name = name

    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.title("Comparaison des Courbes ROC")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n🏆 Le meilleur modèle est : {best_model_name} avec un score de {best_score:.3f}")
    return best_model_obj, results


# ------------------------------------------------------------
# 2. NORMALISATION (Q3)
# ------------------------------------------------------------

def run_normalization_and_compare(X_train, X_test, y_train, y_test, classifiers):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "="*50)
    print("COMPARAISON APRÈS NORMALISATION")
    print("="*50)

    best, scores = run_classifiers_train_test(
        classifiers, X_train_scaled, y_train, X_test_scaled, y_test
    )
    return X_train_scaled, X_test_scaled


# ------------------------------------------------------------
# 3. ACP + CONCAT (Q4)
# ------------------------------------------------------------

def enrichir_donnees(X_train_sc, X_test_sc):
    print("--- Calcul ACP et Concaténation ---")
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    X_train_final = np.hstack([X_train_sc, X_train_pca])
    X_test_final = np.hstack([X_test_sc, X_test_pca])
    print(f"✅ Dimensions après concaténation : {X_train_final.shape} (16 variables)")
    return X_train_final, X_test_final


# ------------------------------------------------------------
# 4. SÉLECTION DE VARIABLES (Q5)
# ------------------------------------------------------------

def calcul_importance_variables(Xtrain, Ytrain, nom_cols):
    print("\n--- 1. Importance (RF) ---")
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(Xtrain, Ytrain)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("Top 5 variables :", nom_cols[sorted_idx[:5]])

    padding = np.arange(Xtrain.shape[1]) + 0.5
    plt.figure(figsize=(10, 8))
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, nom_cols[sorted_idx])
    plt.xlabel("Importance Relative")
    plt.title("Importance (avec ACP)")
    plt.gca().invert_yaxis()
    plt.show()
    return sorted_idx


def selection_nombre_optimal(Xtrain, Ytrain, Xtest, Ytest, sorted_idx):
    print("\n--- 2. Sélection Optimale (MLP) ---")
    model = MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=500)
    scores = np.zeros(Xtrain.shape[1])

    for f in range(Xtrain.shape[1]):
        cols = sorted_idx[:f+1]
        model.fit(Xtrain[:, cols], Ytrain)
        y_pred = model.predict(Xtest[:, cols])
        scores[f] = np.round(accuracy_score(Ytest, y_pred), 3)

    plt.figure(figsize=(8, 4))
    plt.plot(scores, marker='o')
    plt.title("Evolution Accuracy MLP")
    plt.grid(True)
    plt.show()

    best_k = np.argmax(scores) + 1
    print(f"👉 Nombre optimal trouvé : {best_k}")
    return best_k


def explicabilite_shap(Xtrain, Ytrain, nom_cols):
    print("\n--- 3. Explicabilité (SHAP) ---")
    try:
        import shap
        shap.initjs()
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
        model.fit(Xtrain, Ytrain)
        explainer = shap.Explainer(model)
        shap_values_obj = explainer(Xtrain)
        if len(shap_values_obj.shape) == 3:
            vals = shap_values_obj[:, :, 1]
        else:
            vals = shap_values_obj
        shap.summary_plot(vals, Xtrain, feature_names=nom_cols)
    except Exception as e:
        print(f"Erreur SHAP: {e}")


# ------------------------------------------------------------
# 5. TUNING DU MLP (Q6)
# ------------------------------------------------------------

def optimize_best_model_mlp(X_train_full, y_train, sorted_idx, best_k):
    print(f"\n--- Optimisation du MLP sur les {best_k} meilleures variables ---")
    cols_to_keep = sorted_idx[:best_k]
    X_train_opt = X_train_full[:, cols_to_keep]

    param_grid = {
        "hidden_layer_sizes": [(40, 20), (60, 30)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "solver": ["adam", "lbfgs"]
    }

    my_scorer = make_scorer(
        lambda y_true, y_pred: (
            accuracy_score(y_true, y_pred) +
            precision_score(y_true, y_pred, zero_division=0)
        ) / 2
    )

    base_mlp = MLPClassifier(max_iter=1000, random_state=1)

    grid = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=5,
        scoring=my_scorer,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_opt, y_train)

    print(f"✅ Meilleurs paramètres trouvés : {grid.best_params_}")
    print(f"✅ Meilleur score de validation (CV) : {grid.best_score_:.3f}")

    return grid.best_estimator_, cols_to_keep


# ------------------------------------------------------------
# 6. PIPELINE STYLE PROF (Q7)
# ------------------------------------------------------------

def create_and_save_pipeline(best_mlp, best_k, X_train, y_train, pickle_name="credit_scoring.pkl"):
    print("\n--- Création du Pipeline de Production  ---")
    print(f"Nombre de variables sélectionnées (best_k) : {best_k}")

    best_params = best_mlp.get_params()
    final_mlp = MLPClassifier(**best_params)

    P = Pipeline([
        ('S', StandardScaler()),
        ('FU', FeatureUnion([
            ('ID', FunctionTransformer()),
            ('PCA', PCA(n_components=3))
        ])),
        ('RFE', RFE(
            estimator=RandomForestClassifier(n_estimators=1000, random_state=1),
            n_features_to_select=best_k
        )),
        ('MLP', final_mlp)
    ])

    P.fit(X_train, y_train)

    with open(pickle_name, "wb") as f:
        pickle.dump(P, f)

    print(f"✅ Pipeline sauvegardé sous '{pickle_name}'")
    return P


# ------------------------------------------------------------
# 7. ORCHESTRATION (Q8)
# ------------------------------------------------------------

def pipeline_generation_train_test_split(filename="credit_scoring.csv"):
    """
    Orchestration :
      - chargement + split
      - comparaison brut / normalisé / ACP
      - importance + sélection
      - tuning MLP
      - création + sauvegarde du pipeline
    Retourne le pipeline final.
    """
    print("\n=========== ÉTAPE 0 : Chargement & Split ===========")
    X_train, X_test, y_train, y_test, df = load_and_split_data(filename)

    # dictionnaire de classifieurs (comme dans le notebook)
    clfs = {
        'CART': DecisionTreeClassifier(random_state=1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1)
    }

    print("\n=========== ÉTAPE 1 : Données BRUTES ===========")
    _ , _ = run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test)

    print("\n=========== ÉTAPE 2 : Données NORMALISÉES ===========")
    X_train_scaled, X_test_scaled = run_normalization_and_compare(
        X_train, X_test, y_train, y_test, clfs
    )

    print("\n=========== ÉTAPE 3 : NORMALISÉ + ACP (Hybride) ===========")
    X_train_final, X_test_final = enrichir_donnees(X_train_scaled, X_test_scaled)

    feature_names = df.columns[:-1]
    nom_cols_final = np.append(feature_names, ["PCA1", "PCA2", "PCA3"])

    print("\n=========== ÉTAPE 4 : IMPORTANCE & SÉLECTION ===========")
    sorted_indices = calcul_importance_variables(X_train_final, y_train, nom_cols_final)
    best_k = selection_nombre_optimal(X_train_final, y_train, X_test_final, y_test, sorted_indices)
    explicabilite_shap(X_train_final, y_train, nom_cols_final)

    print("\n=========== ÉTAPE 5 : TUNING DU MLP ===========")
    best_mlp_optimized, features_indices = optimize_best_model_mlp(
        X_train_final, y_train, sorted_indices, best_k
    )

    print("\n=========== ÉTAPE 6 : PIPELINE FINAL ===========")
    final_pipeline = create_and_save_pipeline(
        best_mlp_optimized, best_k, X_train, y_train
    )

    print("\n=========== ORCHESTRATION TERMINÉE ✅ ===========")
    return final_pipeline
