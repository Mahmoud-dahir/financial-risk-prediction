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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
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

# ============================================================
# Q1 - Chargement des donnees + split train/test (50/50)
# ============================================================

def load_and_split_data(filename, sep=";"):
    """
    Charge le fichier credit_scoring.csv, affiche quelques stats simples,
    puis fait un split 50/50 (train/test) avec random_state=1.
    """
    try:
        df = pd.read_csv(filename, sep=sep)
    except Exception:
        df = pd.read_csv(filename, sep=",")

    data = df.values
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {n_pos} positifs ({n_pos/len(y):.2%}) / {n_neg} negatifs")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


# ============================================================
# Q2 - Evaluation + comparaison des classifieurs
# ============================================================

def evaluation(model, X_test, y_test, model_name, business_choice="precision"):
    """
    Affiche:
      - matrice de confusion
      - accuracy / precision / recall
      - ROC + AUC
    Retourne un score final = (accuracy + critere_metier) / 2

    business_choice:
      - "precision" (par defaut)
      - "recall"
      - "best" : max(precision, recall) (si tu veux coller a "meilleur des deux")
    """
    y_pred = model.predict(X_test)

    # Score continu pour ROC/AUC (proba si dispo, sinon decision_function)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred  # fallback

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")

    if business_choice == "recall":
        crit = rec
        crit_name = "Recall"
    elif business_choice == "best":
        crit = max(prec, rec)
        crit_name = "Best(Prec/Rec)"
    else:
        crit = prec
        crit_name = "Precision"

    score_final = (acc + crit) / 2

    print(f"\n--- {model_name} ---")
    print("Confusion matrix:")
    print(cm)
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print(f"AUC:      {auc:.3f}")
    print(f"Final score = (Accuracy + {crit_name})/2 : {score_final:.3f}")

    # Courbe ROC (si possible)
    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
    except Exception:
        pass

    return score_final


def run_classifiers_train_test(classifiers, X_train, y_train, X_test, y_test, business_choice="precision"):
    """
    Entraine chaque classifieur du dictionnaire "classifiers", evalue sur le test,
    trace les ROC et renvoie le meilleur modele selon le score final.
    """
    results = {}
    best_score = -np.inf
    best_model_obj = None
    best_model_name = None

    plt.figure(figsize=(8, 6))

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        score = evaluation(model, X_test, y_test, name, business_choice=business_choice)
        results[name] = score

        if score > best_score:
            best_score = score
            best_model_obj = model
            best_model_name = name

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC curves comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest model: {best_model_name} | best score: {best_score:.3f}")
    return best_model_obj, results


# ============================================================
# Q3 - Normalisation (StandardScaler)
# ============================================================

def run_normalization_and_compare(X_train, X_test, y_train, y_test, classifiers, business_choice="precision"):
    """
    Normalisation standard: fit sur train, transform sur test.
    Puis comparaison des classifieurs.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 50)
    print("COMPARISON AFTER STANDARDIZATION")
    print("=" * 50)

    run_classifiers_train_test(
        classifiers, X_train_scaled, y_train, X_test_scaled, y_test, business_choice=business_choice
    )
    return X_train_scaled, X_test_scaled


# ============================================================
# Q4 - ACP (3 composantes) + concat aux features normalisees
# ============================================================

def enrichir_donnees(X_train_sc, X_test_sc):
    """
    ACP sur train, transform sur test, puis concatene les 3 composantes
    aux features normalisees.
    """
    print("PCA computation + concatenation")
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    X_train_final = np.hstack([X_train_sc, X_train_pca])
    X_test_final = np.hstack([X_test_sc, X_test_pca])

    print(f"New train shape after concat: {X_train_final.shape}")
    return X_train_final, X_test_final


# ============================================================
# Q5 - Selection de variables (RF importance + nb optimal + SHAP)
# ============================================================

def calcul_importance_variables(Xtrain, Ytrain, nom_cols):
    """
    Importance des variables via RandomForest (n_estimators=1000).
    Conforme a l'enonce: on calcule aussi std et on l'affiche en xerr.
    """
    print("\n[Q5.1] RandomForest feature importances")

    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(Xtrain, Ytrain)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    sorted_idx = np.argsort(importances)[::-1]

    print("Features sorted by importance:")
    print(nom_cols[sorted_idx])

    padding = np.arange(Xtrain.shape[1]) + 0.5
    plt.figure(figsize=(10, 8))
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align="center")
    plt.yticks(padding, nom_cols[sorted_idx])
    plt.xlabel("Relative importance")
    plt.title("Variable importance (RF)")
    plt.gca().invert_yaxis()
    plt.show()

    return sorted_idx


def selection_nombre_optimal(Xtrain, Ytrain, Xtest, Ytest, sorted_idx):
    """
    Choix du nombre de variables a garder.
    Ici on suit ton approche (MLP) et on regarde l'evolution de l'accuracy
    quand on ajoute les variables les plus importantes une par une.
    """
    print("\n[Q5.2] Choosing the number of features (accuracy curve)")

    model = MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=500)
    scores = np.zeros(Xtrain.shape[1])

    for f in range(Xtrain.shape[1]):
        cols = sorted_idx[: f + 1]
        model.fit(Xtrain[:, cols], Ytrain)
        y_pred = model.predict(Xtest[:, cols])
        scores[f] = np.round(accuracy_score(Ytest, y_pred), 3)

    plt.figure(figsize=(8, 4))
    plt.plot(scores, marker="o")
    plt.title("Accuracy vs number of selected features")
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    best_k = int(np.argmax(scores) + 1)
    print(f"Best_k found: {best_k}")
    return best_k


def explicabilite_shap(Xtrain, Ytrain, nom_cols):
    """
    Explicabilite globale avec SHAP (si la lib est installee).
    """
    print("\n[Q5.3] SHAP explainability (global)")

    try:
        import shap

        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
        model.fit(Xtrain, Ytrain)

        explainer = shap.Explainer(model)
        shap_values_obj = explainer(Xtrain)

        # Selon la version SHAP / modele, la forme peut varier
        if len(shap_values_obj.shape) == 3:
            vals = shap_values_obj[:, :, 1]
        else:
            vals = shap_values_obj

        shap.summary_plot(vals, Xtrain, feature_names=nom_cols)
    except Exception as e:
        print(f"SHAP not available or failed: {e}")


# ============================================================
# Q6 - Tuning (GridSearchCV) sur (accuracy+precision)/2
# ============================================================

def optimize_best_model_mlp(X_train_full, y_train, sorted_idx, best_k):
    """
    Optimise un MLP via GridSearchCV, en gardant les best_k variables
    (selon sorted_idx). Le score utilise (accuracy+precision)/2.
    """
    print(f"\n[Q6] GridSearchCV on MLP (top {best_k} features)")

    cols_to_keep = sorted_idx[:best_k]
    X_train_opt = X_train_full[:, cols_to_keep]

    param_grid = {
        "hidden_layer_sizes": [(40, 20), (60, 30)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "solver": ["adam", "lbfgs"],
    }

    my_scorer = make_scorer(
        lambda y_true, y_pred: (
            accuracy_score(y_true, y_pred)
            + precision_score(y_true, y_pred, zero_division=0)
        )
        / 2
    )

    base_mlp = MLPClassifier(max_iter=1000, random_state=1)

    grid = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=5,
        scoring=my_scorer,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_opt, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", round(grid.best_score_, 3))

    return grid.best_estimator_, cols_to_keep


# ============================================================
# Q7 - Pipeline final + sauvegarde pickle (train/test)
# ============================================================

def create_and_save_pipeline(best_mlp, best_k, X_train, y_train, pickle_name="credit_scoring.pkl"):
    """
    Pipeline de production (train/test):
      - StandardScaler
      - FeatureUnion (ID + PCA(3))  -> concat features normalisees + 3 composantes
      - RFE(RandomForest)          -> selection de best_k variables
      - MLP (avec meilleurs params)
    """
    print("\n[Q7] Building the final pipeline (train/test)")

    final_mlp = MLPClassifier(**best_mlp.get_params())

    P = Pipeline(
        [
            ("S", StandardScaler()),
            (
                "FU",
                FeatureUnion(
                    [
                        ("ID", FunctionTransformer()),
                        ("PCA", PCA(n_components=3)),
                    ]
                ),
            ),
            (
                "RFE",
                RFE(
                    estimator=RandomForestClassifier(n_estimators=1000, random_state=1),
                    n_features_to_select=best_k,
                ),
            ),
            ("MLP", final_mlp),
        ]
    )

    P.fit(X_train, y_train)

    with open(pickle_name, "wb") as f:
        pickle.dump(P, f)

    print(f"Pipeline saved to: {pickle_name}")
    return P


# ============================================================
# Q8 - Orchestration train/test split
# ============================================================

def pipeline_generation_train_test_split(filename="credit_scoring.csv", business_choice="precision"):
    """
    Orchestration Q1 -> Q8:
      - load/split
      - comparaison brut / normalise / normalise+ACP
      - importance + choix best_k + shap
      - tuning
      - pipeline final + pickle
    """
    print("\n=========== STEP 0: Load & split ===========")
    X_train, X_test, y_train, y_test, df = load_and_split_data(filename)

    clfs = {
        "CART": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1),
    }

    print("\n=========== STEP 1: Raw data ===========")
    run_classifiers_train_test(clfs, X_train, y_train, X_test, y_test, business_choice=business_choice)

    print("\n=========== STEP 2: Standardized data ===========")
    X_train_scaled, X_test_scaled = run_normalization_and_compare(
        X_train, X_test, y_train, y_test, clfs, business_choice=business_choice
    )

    print("\n=========== STEP 3: Standardized + PCA concat ===========")
    X_train_final, X_test_final = enrichir_donnees(X_train_scaled, X_test_scaled)

    print("\nComparison on standardized + PCA-concat features")
    run_classifiers_train_test(clfs, X_train_final, y_train, X_test_final, y_test, business_choice=business_choice)

    feature_names = df.columns[:-1]
    nom_cols_final = np.append(feature_names, ["PCA1", "PCA2", "PCA3"])

    print("\n=========== STEP 4: Feature selection (importance + best_k + shap) ===========")
    sorted_indices = calcul_importance_variables(X_train_final, y_train, nom_cols_final)
    best_k = selection_nombre_optimal(X_train_final, y_train, X_test_final, y_test, sorted_indices)
    explicabilite_shap(X_train_final, y_train, nom_cols_final)

    print("\n=========== STEP 5: Tuning ===========")
    best_mlp_optimized, _ = optimize_best_model_mlp(X_train_final, y_train, sorted_indices, best_k)

    print("\n=========== STEP 6: Final pipeline + pickle ===========")
    final_pipeline = create_and_save_pipeline(best_mlp_optimized, best_k, X_train, y_train)

    print("\nOrchestration finished.")
    return final_pipeline


# ============================================================
# Q10 - Validation croisee (sans fuite de donnees)
# ============================================================

def build_cv_classifiers():
    """
    Dictionnaire des classifieurs a comparer en CV (Q10).
    """
    clfs_cv = {
        "CART": DecisionTreeClassifier(criterion="gini", random_state=1),
        "ID3": DecisionTreeClassifier(criterion="entropy", random_state=1),
        "Stump": DecisionTreeClassifier(max_depth=1, random_state=1),
        "MLP_20_10": MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=1),
        "KNN_5": KNeighborsClassifier(n_neighbors=5),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=1),
            n_estimators=200,
            random_state=1,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=1),
        "RF_200": RandomForestClassifier(n_estimators=200, random_state=1),
    }

    # XGBoost si disponible (optionnel)
    try:
        from xgboost import XGBClassifier

        clfs_cv["XGB_200"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=1,
        )
    except Exception:
        print("XGBoost not installed, continuing without it.")

    return clfs_cv


def prepare_dataset_raw_for_cv(filename="credit_scoring.csv", sep=";"):
    """
    Pour la CV, on charge uniquement X_raw et y.
    Important: normalisation et ACP seront faites dans le Pipeline pendant la CV,
    pour respecter la methode (fit sur train du fold, transform sur test du fold).
    """
    try:
        df = pd.read_csv(filename, sep=sep)
    except Exception:
        df = pd.read_csv(filename, sep=",")

    data = df.values
    X_raw = data[:, :-1].astype(float)
    y = data[:, -1].astype(float)

    print(f"Raw dataset shape: {X_raw.shape}")
    return X_raw, y, df.columns[:-1]


def _preprocess_steps(mode):
    """
    Construit les etapes de preprocessing pour la CV:
      - "raw"  : aucune transformation
      - "norm" : StandardScaler
      - "acp"  : StandardScaler + FeatureUnion(ID + PCA(3)) pour concatener les 3 composantes
    """
    if mode == "raw":
        return []
    if mode == "norm":
        return [("S", StandardScaler())]
    if mode == "acp":
        return [
            ("S", StandardScaler()),
            ("FU", FeatureUnion([("ID", FunctionTransformer()), ("PCA", PCA(n_components=3))])),
        ]
    raise ValueError("mode must be in {'raw','norm','acp'}")


def run_classifiers_cv(clfs, X_raw, y, n_splits=10, mode="raw"):
    """
    Compare plusieurs classifieurs via CV 10-fold et affiche:
      - accuracy (moyenne +/- std)
      - AUC (moyenne +/- std)
      - score metier (accuracy+precision)/2 (moyenne +/- std)
      - temps d'execution

    Le preprocessing est fait dans un Pipeline pour eviter la fuite de donnees.
    """
    print("\n===== CROSS-VALIDATION =====")
    print(f"Mode: {mode} | X_raw shape (before preprocessing): {X_raw.shape}")
    if mode == "acp":
        print("Note: PCA concat happens inside the pipeline (13 -> 16 features per fold).")


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "score_metier": make_scorer(
            lambda y_true, y_pred: (
                accuracy_score(y_true, y_pred)
                + precision_score(y_true, y_pred, zero_division=0)
            )
            / 2
        ),
    }

    prep = _preprocess_steps(mode)
    results = {}

    for name, clf in clfs.items():
        print(f"\n--- {name} ---")

        # Pipeline = preprocessing (eventuel) + classifieur
        model = Pipeline(prep + [("CLF", clf)])

        start = time.time()
        cv_res = cross_validate(
            model,
            X_raw,
            y,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        delta = time.time() - start

        acc_mean = cv_res["test_accuracy"].mean()
        acc_std = cv_res["test_accuracy"].std()

        auc_mean = cv_res["test_roc_auc"].mean()
        auc_std = cv_res["test_roc_auc"].std()

        met_mean = cv_res["test_score_metier"].mean()
        met_std = cv_res["test_score_metier"].std()

        print(f"Accuracy     : {acc_mean:.3f} +/- {acc_std:.3f}")
        print(f"AUC          : {auc_mean:.3f} +/- {auc_std:.3f}")
        print(f"Score metier : {met_mean:.3f} +/- {met_std:.3f}")
        print(f"Time (s)     : {delta:.2f}")

        results[name] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "score_metier_mean": met_mean,
            "score_metier_std": met_std,
            "time_sec": delta,
        }

    return results


def tune_best_model(best_name, X_raw, y, mode="acp"):
    """
    Tuning du meilleur modele apres la CV.
    Important: on tune un Pipeline complet (preprocessing + modele) pour eviter la fuite de donnees.
    On retourne ensuite uniquement le classifieur (pas le pipeline), car il sera reintegre dans le pipeline final.
    """
    my_scorer = make_scorer(
        lambda yt, yp: (accuracy_score(yt, yp) + precision_score(yt, yp, zero_division=0)) / 2
    )

    # Preprocessing identique a ce qu'on a utilise pour choisir le best model
    prep = _preprocess_steps(mode)
    base_clf = None
    param_grid = None

    if best_name.startswith("XGB"):
        from xgboost import XGBClassifier

        base_clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=1,
        )
        param_grid = {
            "CLF__n_estimators": [200, 400],
            "CLF__max_depth": [3, 4, 6],
            "CLF__learning_rate": [0.05, 0.1],
            "CLF__subsample": [0.8, 1.0],
            "CLF__colsample_bytree": [0.8, 1.0],
        }

    elif best_name.startswith("RF"):
        base_clf = RandomForestClassifier(random_state=1)
        param_grid = {
            "CLF__n_estimators": [200, 400],
            "CLF__max_depth": [None, 10, 20],
            "CLF__min_samples_split": [2, 5],
            "CLF__min_samples_leaf": [1, 2],
        }

    else:
        # Si pas de grille prevue, on garde simplement le modele tel quel
        print(f"No tuning grid for {best_name}. Keeping default parameters for this model.")
        return build_cv_classifiers()[best_name]

    pipe = Pipeline(prep + [("CLF", base_clf)])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring=my_scorer,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_raw, y)

    print("Best params:", grid.best_params_)
    print("Best tuning score (5-fold CV):", round(grid.best_score_, 3))

    # On renvoie uniquement le classifieur optimise
    best_pipe = grid.best_estimator_
    return best_pipe.named_steps["CLF"]


def create_and_save_cv_pipeline(best_estimator, k_features, X_raw, y, pickle_name="credit_scoring_cv.pkl"):
    """
    Pipeline final (version CV) a mettre "en prod":
      - StandardScaler
      - FeatureUnion (ID + PCA(3))  -> concat des 3 composantes
      - SelectKBest                 -> selection de k_features
      - classifieur final
    """
    final_pipeline = Pipeline(
        [
            ("S", StandardScaler()),
            ("FU", FeatureUnion([("ID", FunctionTransformer()), ("PCA", PCA(n_components=3))])),
            ("SKB", SelectKBest(score_func=f_classif, k=k_features)),
            ("CLF", best_estimator),
        ]
    )

    final_pipeline.fit(X_raw, y)

    with open(pickle_name, "wb") as f:
        pickle.dump(final_pipeline, f)

    print(f"CV pipeline saved to: {pickle_name}")
    return final_pipeline


def pipeline_generation_cv(filename="credit_scoring.csv"):
    """
    Orchestration Q10:
      - CV sur raw / norm / acp (sans fuite)
      - choix du meilleur algo sur le cas "acp"
      - tuning du meilleur algo (sans fuite)
      - creation + sauvegarde du pipeline final
    """
    print("\n=========== CV STEP 0: Load raw dataset ===========")
    X_raw, y, _ = prepare_dataset_raw_for_cv(filename)

    print("\n=========== CV STEP 1: Build classifiers dict ===========")
    clfs_cv = build_cv_classifiers()

    print("\n=========== CV STEP 2: CV on raw data ===========")
    res_raw = run_classifiers_cv(clfs_cv, X_raw, y, mode="raw")

    print("\n=========== CV STEP 3: CV on standardized data ===========")
    res_norm = run_classifiers_cv(clfs_cv, X_raw, y, mode="norm")

    print("\n=========== CV STEP 4: CV on standardized + PCA-concat data ===========")
    res_acp = run_classifiers_cv(clfs_cv, X_raw, y, mode="acp")

    print("\n=========== CV STEP 5: Pick best model (based on score metier) ===========")
    best_name = None
    best_score = -np.inf

    for name, metrics in res_acp.items():
        if metrics["score_metier_mean"] > best_score:
            best_score = metrics["score_metier_mean"]
            best_name = name

    print(f"Best model on mode='acp': {best_name} | mean score metier: {best_score:.3f}")

    print("\n=========== CV STEP 6: Tuning + final pipeline ===========")

    best_estimator = tune_best_model(best_name, X_raw, y, mode="acp")

    # k_features: a adapter selon tes choix (enonce: selection de variables)
    k_features = 13

    final_pipeline = create_and_save_cv_pipeline(
        best_estimator=best_estimator,
        k_features=k_features,
        X_raw=X_raw,
        y=y,
        pickle_name="credit_scoring_cv.pkl",
    )

    print("CV orchestration finished.")
    return final_pipeline, best_name, best_score, res_acp
