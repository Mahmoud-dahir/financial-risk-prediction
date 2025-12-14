import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    make_scorer
)

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=10000, suppress=True)


# ------------------------------------------------------------
# 0. CHARGEMENT ET SPLIT (Q1)
# ------------------------------------------------------------

def load_and_split_data(filename, sep=';'):
    """
    Charge le fichier, affiche quelques stats, et fait un split 50/50
    avec stratification.
    Retourne : X_train, X_test, y_train, y_test, df
    """
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
    """
    Affiche confusion, accuracy, précision, rappel, AUC,
    et ajoute la courbe ROC sur la figure courante.
    Retourne le score métier (acc + préc) / 2.
    """
    y_pred = model.predict(X_test)

    # ✅ score continu pour ROC/AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred  # fallback (moins bon, mais évite de casser)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")

    score_final = (acc + prec) / 2

    print(f"\n--- {model_name} ---")
    print(f"Matrice :\n{cm}")
    print(f"Accuracy : {acc:.3f} | Précision : {prec:.3f} | Rappel : {rec:.3f}")
    print(f"AUC      : {auc:.3f}")
    print(f"👉 Score Final : {score_final:.3f}")

    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
    except Exception:
        pass

    return score_final



def run_classifiers_train_test(classifiers, X_train, y_train, X_test, y_test):
    """
    Entraîne chaque classifieur du dict 'classifiers' et les évalue.
    Affiche les courbes ROC et retourne :
      - le meilleur modèle
      - le dict des scores
    """
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
    """
    Applique StandardScaler (fit sur train uniquement),
    puis relance la comparaison des classifieurs.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "="*50)
    print("COMPARAISON APRÈS NORMALISATION")
    print("="*50)

    _best, _scores = run_classifiers_train_test(
        classifiers, X_train_scaled, y_train, X_test_scaled, y_test
    )
    return X_train_scaled, X_test_scaled


# ------------------------------------------------------------
# 3. ACP + CONCAT (Q4)
# ------------------------------------------------------------

def enrichir_donnees(X_train_sc, X_test_sc):
    """
    Calcule une ACP (3 composantes) sur les données normalisées,
    puis concatène les 3 composantes aux features initiales.
    """
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
    """
    Utilise un RandomForest pour calculer l'importance des variables
    et affiche un barplot trié.
    """
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
    """
    Teste un MLP sur les f meilleures variables (f=1..p)
    et retourne le nombre optimal best_k.
    """
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
    """
    Explique un RandomForest via SHAP (globale).
    """
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
    """
    Sélectionne les best_k meilleures variables (sur X_train_full)
    et optimise le MLP via GridSearchCV avec le critère (acc+prec)/2.
    """
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
    """
    Crée un pipeline :
      - StandardScaler
      - FeatureUnion(ID, PCA(3))
      - RFE(RandomForest, n_features=best_k)
      - MLP optimisé
    Entraîne sur X_train / y_train (13 features brutes) et sauvegarde.
    """
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
# 7. ORCHESTRATION TRAIN/TEST SPLIT (Q8)
# ------------------------------------------------------------

def pipeline_generation_train_test_split(filename="credit_scoring.csv"):
    """
    Orchestration 1 :
      - chargement + split
      - comparaison brut / normalisé / ACP
      - importance + sélection
      - tuning MLP
      - création + sauvegarde du pipeline
    Retourne le pipeline final.
    """
    print("\n=========== ÉTAPE 0 : Chargement & Split ===========")
    X_train, X_test, y_train, y_test, df = load_and_split_data(filename)

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

    print("\n" + "="*50)
    print("COMPARAISON SUR DONNÉES NORMALISÉES + ACP (Stratégie Hybride)")
    print("="*50)

    _, _ = run_classifiers_train_test(clfs, X_train_final, y_train, X_test_final, y_test)

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


# ------------------------------------------------------------
# 8. PARTIE VALIDATION CROISÉE (Q10)
# ------------------------------------------------------------

def build_cv_classifiers():
    """
    Construit un dictionnaire de classifieurs pour la validation croisée.
    """
    clfs_cv = {
        'CART': DecisionTreeClassifier(criterion="gini", random_state=1),
        'ID3': DecisionTreeClassifier(criterion="entropy", random_state=1),
        'Stump': DecisionTreeClassifier(max_depth=1, random_state=1),
        'MLP_20_10': MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=1),
        'KNN_5': KNeighborsClassifier(n_neighbors=5),
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=1),
            n_estimators=200,
            random_state=1
        ),

        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            random_state=1
        ),
        'RF_200': RandomForestClassifier(
            n_estimators=200,
            random_state=1
        )
    }

    # Optionnel : XGBoost si dispo
    try:
        from xgboost import XGBClassifier
        clfs_cv['XGB_200'] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=1
        )
    except Exception:
        print("⚠️ XGBoost non disponible, on continue sans lui.")

    return clfs_cv


def prepare_datasets_for_cv(filename="credit_scoring.csv", sep=';'):
    """
    Prépare les 3 jeux pour la CV :
      - X_raw  : brut
      - X_norm : normalisé
      - X_acp  : normalisé + ACP (3 dims)
    """
    try:
        df = pd.read_csv(filename, sep=sep)
    except:
        df = pd.read_csv(filename, sep=',')

    data = df.values
    X_raw = data[:, :-1].astype(float)
    y = data[:, -1].astype(float)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)
    X_acp = np.hstack([X_norm, X_pca])

    feature_names = df.columns[:-1]
    nom_cols_final = np.append(feature_names, ["PCA1", "PCA2", "PCA3"])

    print(f"Jeu original   : {X_raw.shape}")
    print(f"Jeu normalisé  : {X_norm.shape}")
    print(f"Jeu norm+ACP   : {X_acp.shape}")

    return X_raw, X_norm, X_acp, y, feature_names, nom_cols_final


def run_classifiers_cv(clfs, X, y, n_splits=10):
    """
    Compare plusieurs classifieurs via validation croisée KFold.
    Pour chaque classifieur :
      - Accuracy (moy ± std)
      - AUC (moy ± std)
      - Score métier (acc+prec)/2 (moy ± std)
      - Temps d'exécution
    """
    print("\n===== VALIDATION CROISÉE SUR CE JEU DE DONNÉES =====")
    print(f"X shape : {X.shape}, y shape : {y.shape}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'score_metier': make_scorer(
            lambda y_true, y_pred: (
                accuracy_score(y_true, y_pred) +
                precision_score(y_true, y_pred, zero_division=0)
            ) / 2
        )
    }

    results = {}

    for name, clf in clfs.items():
        print(f"\n--- {name} ---")
        start = time.time()
        cv_res = cross_validate(
            clf,
            X,
            y,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        end = time.time()
        delta = end - start

        acc_mean = cv_res['test_accuracy'].mean()
        acc_std = cv_res['test_accuracy'].std()

        auc_mean = cv_res['test_roc_auc'].mean()
        auc_std = cv_res['test_roc_auc'].std()

        met_mean = cv_res['test_score_metier'].mean()
        met_std = cv_res['test_score_metier'].std()

        print(f"Accuracy        : {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"AUC             : {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"Score métier    : {met_mean:.3f} ± {met_std:.3f}")
        print(f"Temps (s)       : {delta:.2f}")

        results[name] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "score_metier_mean": met_mean,
            "score_metier_std": met_std,
            "time_sec": delta
        }

    return results

def tune_best_model(best_name, X, y):
    """
    Tuning du meilleur modèle trouvé en CV.
    Retourne best_estimator_.
    """
    my_scorer = make_scorer(
        lambda yt, yp: (accuracy_score(yt, yp) + precision_score(yt, yp, zero_division=0)) / 2
    )

    if best_name.startswith("XGB"):
        from xgboost import XGBClassifier
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=1
        )
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

    elif best_name.startswith("RF"):
        base = RandomForestClassifier(random_state=1)
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

    else:
        # fallback : pas de tuning, on renvoie le modèle tel quel
        print("⚠️ Pas de grille de tuning prévue pour", best_name, "=> modèle gardé tel quel.")
        return build_cv_classifiers()[best_name]

    grid = GridSearchCV(
        base,
        param_grid=param_grid,
        cv=5,
        scoring=my_scorer,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)

    print("✅ Best params (tuning) :", grid.best_params_)
    print("✅ Best score tuning (CV 5-fold) :", round(grid.best_score_, 3))
    return grid.best_estimator_


def create_and_save_cv_pipeline(best_estimator, k_features, X_raw, y, pickle_name="credit_scoring_cv.pkl"):
    """
    Pipeline CV final (ENTRAÎNÉ puis sauvegardé) :
      StandardScaler -> FeatureUnion(ID + PCA3) -> SelectKBest -> best_estimator
    """
    final_pipeline = Pipeline([
        ('S', StandardScaler()),
        ('FU', FeatureUnion([
            ('ID', FunctionTransformer()),
            ('PCA', PCA(n_components=3))
        ])),
        ('SKB', SelectKBest(score_func=f_classif, k=k_features)),
        ('CLF', best_estimator)
    ])

    final_pipeline.fit(X_raw, y)

    with open(pickle_name, "wb") as f:
        pickle.dump(final_pipeline, f)

    print(f"✅ Pipeline (version CV) sauvegardé sous '{pickle_name}'")
    return final_pipeline




def pipeline_generation_cv(filename="credit_scoring.csv"):
    print("\n=========== CV ÉTAPE 0 : Préparation des jeux de données ===========")
    X_raw, X_norm, X_acp, y, feature_names, nom_cols_final = prepare_datasets_for_cv(filename)

    print("\n=========== CV ÉTAPE 1 : Construction des classifieurs ===========")
    clfs_cv = build_cv_classifiers()

    print("\n=========== CV ÉTAPE 2 : Comparaison sur DONNÉES ORIGINALES ===========")
    res_raw = run_classifiers_cv(clfs_cv, X_raw, y)

    print("\n=========== CV ÉTAPE 3 : Comparaison sur DONNÉES NORMALISÉES ===========")
    res_norm = run_classifiers_cv(clfs_cv, X_norm, y)

    print("\n=========== CV ÉTAPE 4 : Comparaison sur DONNÉES NORMALISÉES + ACP ===========")
    res_acp = run_classifiers_cv(clfs_cv, X_acp, y)

    print("\n=========== CV ÉTAPE 5 : Sélection du meilleur algorithme ===========")
    best_name = None
    best_score = -np.inf

    for name, metrics in res_acp.items():
        if metrics["score_metier_mean"] > best_score:
            best_score = metrics["score_metier_mean"]
            best_name = name

    print(f"\n🏆 Meilleur algorithme (CV, données norm+ACP) : {best_name} avec score métier moyen = {best_score:.3f}")

    print("\n=========== CV ÉTAPE 6 : Sélection + Tuning + Pipeline final ===========")

    # 1) Tuning du meilleur modèle sur le dataset CV (X_acp)
    best_estimator = tune_best_model(best_name, X_acp, y)

    # 2) SelectKBest s’applique après FeatureUnion => sur 16 features
    k_features = 13  # tu peux mettre 16 si tu veux garder toutes les features

    # 3) Création + fit + sauvegarde du pipeline final
    final_pipeline = create_and_save_cv_pipeline(
        best_estimator=best_estimator,
        k_features=k_features,
        X_raw=X_raw,
        y=y,
        pickle_name="credit_scoring_cv.pkl"
    )

    print("=========== ORCHESTRATION CV TERMINÉE ✅ ===========")
    return final_pipeline, best_name, best_score, res_acp



