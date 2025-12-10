import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# --- CLASSE TECHNIQUE (Nécessaire pour le pipeline hybride) ---
class HybridACP(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
    def fit(self, X, y=None):
        self.scaler.fit(X)
        X_sc = self.scaler.transform(X)
        self.pca.fit(X_sc)
        return self
    def transform(self, X):
        X_sc = self.scaler.transform(X)
        X_pca = self.pca.transform(X_sc)
        return np.hstack([X_sc, X_pca])

# ============================================================
# RÉPONSE AUX CONSIGNES (FONCTIONS NOMMÉES EXACTEMENT)
# ============================================================

# 1. Fonction d'apprentissage et de test
def apprentissage_test_algorithmes(X_train, y_train, X_test, y_test, label="Données"):
    print(f"\n--- Apprentissage et Test sur {label} ---")
    clfs = {
        'CART': DecisionTreeClassifier(random_state=1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=500)
    }
    
    best_score = 0
    best_name = ""

    for name, model in clfs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        score = (acc + prec) / 2
        
        print(f"   -> {name} | Score (Acc+Prec)/2 : {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_name = name
            
    print(f"🏆 Meilleur algo sur cette étape : {best_name}")
    return best_name

# 2. Fonction de préparation (Données avec nouvelles variables)
def preparation_donnees_nouvelles_variables(X_train, X_test):
    print("\n--- Création des nouvelles variables (ACP + Concaténation) ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    
    X_train_final = np.hstack([X_train_sc, X_train_pca])
    X_test_final = np.hstack([X_test_sc, X_test_pca])
    return X_train_final, X_test_final

# 3. Fonction de sélection des variables
def selection_variables_pertinentes(X_train, y_train, X_test, y_test, nom_cols):
    print("\n--- Sélection des variables les plus pertinentes ---")
    
    # A. Classement par Random Forest
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(X_train, y_train)
    sorted_idx = np.argsort(clf.feature_importances_)[::-1]
    print("Top 5 variables :", nom_cols[sorted_idx[:5]])
    
    # B. Nombre optimal par itération MLP
    model = MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=500)
    scores = []
    
    for f in range(X_train.shape[1]):
        cols = sorted_idx[:f+1]
        model.fit(X_train[:, cols], y_train)
        y_pred = model.predict(X_test[:, cols])
        scores.append(accuracy_score(y_test, y_pred))
        
    best_k = np.argmax(scores) + 1
    print(f"👉 Nombre optimal retenu : {best_k}")
    
    return sorted_idx, best_k

# 4. Fonction de recherche des meilleurs paramètres
def recherche_meilleurs_parametres(X_train, y_train, sorted_idx, best_k):
    print(f"\n--- Recherche et Choix des Meilleurs Paramètres (GridSearch) ---")
    
    cols_to_keep = sorted_idx[:best_k]
    X_train_opt = X_train[:, cols_to_keep]
    
    param_grid = {
        'hidden_layer_sizes': [(40, 20), (60, 30), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
        'solver': ['adam']
    }
    
    def custom_score(y, yp): 
        return (accuracy_score(y, yp) + precision_score(y, yp, zero_division=0))/2
    
    grid = GridSearchCV(
        MLPClassifier(max_iter=800, random_state=1),
        param_grid,
        scoring=make_scorer(custom_score),
        cv=5, n_jobs=-1, verbose=1
    )
    
    grid.fit(X_train_opt, y_train)
    print(f"✅ Meilleurs paramètres : {grid.best_params_}")
    
    return grid.best_estimator_, cols_to_keep

# 5. Fonction de création du pipeline
def creation_pipeline_apprentissage(X_train_raw, y_train, features_indices, best_model, filename="credit_scoring.pkl"):
    print("\n--- Création du Pipeline d'Apprentissage ---")
    
    params = best_model.get_params()
    clean_params = {k:v for k,v in params.items() 
                    if k in ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate_init', 'max_iter', 'random_state']}
    
    selector = ColumnTransformer(
        [('select_best', 'passthrough', features_indices)], 
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('augmentation', HybridACP()),
        ('selection', selector),
        ('mlp', MLPClassifier(**clean_params))
    ])
    
    pipeline.fit(X_train_raw, y_train)
    
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"✅ Pipeline sauvegardé dans {filename}")
    return pipeline
