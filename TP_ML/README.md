# CreditScore — Application de credit scoring

Application de **classification de dossiers de crédit** construite avec Python, scikit-learn et FastAPI.

Le projet couvre toute la chaîne d’un projet de machine learning :

- préparation et analyse des données ;
- comparaison de plusieurs modèles de classification ;
- normalisation et réduction de dimension par ACP/PCA ;
- sélection de variables ;
- optimisation des hyperparamètres ;
- sauvegarde d’un pipeline entraîné ;
- exposition du modèle avec une API FastAPI ;
- interface web permettant de tester une prédiction.

> **Important :** le résultat est un score produit par un modèle statistique. Il ne constitue ni une garantie de remboursement ni une décision bancaire automatique.

## Fonctionnalités

- Saisie des 13 caractéristiques d’un dossier de crédit.
- Envoi des données à l’endpoint `POST /predict/`.
- Prédiction d’une classe binaire `0` ou `1`.
- Affichage des probabilités associées aux deux classes.
- Vérification du chargement du modèle avec `GET /health`.
- Documentation interactive de l’API avec Swagger.
- Interface web responsive accessible depuis `/app`.

## Aperçu de l’architecture

```text
Interface HTML / CSS / JavaScript
              │
              │ requête JSON
              ▼
        API FastAPI
              │
              ▼
 Pipeline scikit-learn sauvegardé
              │
              ▼
 Classe prédite + probabilités
```

## Modèle de machine learning

Le fichier utilisé actuellement par l’API est :

```text
credit_scoring.pkl
```

Le pipeline sauvegardé regroupe les étapes suivantes :

```text
StandardScaler
→ FeatureUnion : variables standardisées + PCA(3)
→ RFE : sélection de 13 variables
→ MLPClassifier
```

Le notebook compare notamment :

- CART / arbre de décision ;
- K plus proches voisins ;
- réseau de neurones MLP ;
- Random Forest ;
- Bagging ;
- AdaBoost ;
- XGBoost lorsqu’il est installé.

Une seconde version issue de la validation croisée est disponible dans :

```text
credit_scoring_cv.pkl
```

L’API charge volontairement `credit_scoring.pkl`. Pour utiliser l’autre pipeline, il faut modifier `MODEL_PATH` dans `api.py` et vérifier que les dépendances nécessaires sont installées.

## Données

Le jeu `credit_scoring.csv` contient :

- **4 375 dossiers** ;
- **13 variables explicatives** ;
- une cible binaire appelée `Status`.

Variables utilisées par l’API :

| Variable | Description générale |
| --- | --- |
| `Seniority` | Ancienneté |
| `Home` | Situation de logement encodée |
| `Time` | Durée du crédit |
| `Age` | Âge |
| `Marital` | Situation familiale encodée |
| `Records` | Historique encodé |
| `Job` | Situation professionnelle encodée |
| `Expenses` | Dépenses |
| `Income` | Revenus |
| `Assets` | Actifs |
| `Debt` | Dette existante |
| `Amount` | Montant demandé |
| `Price` | Prix du bien financé |

Certaines variables catégorielles sont encore affichées sous forme de codes (`0`, `1` ou `2`). Le dépôt ne fournit pas de dictionnaire métier suffisamment explicite pour remplacer ces valeurs par des libellés sans risque d’erreur.

## Interprétation du résultat

Une réponse comme :

```json
{
  "prediction": 1,
  "proba_classe_0": 0.1886,
  "proba_classe_1": 0.8114
}
```

signifie que le modèle attribue environ **81,14 % de son score probabiliste à la classe 1**.

Cela ne signifie pas automatiquement :

> « Cette personne a exactement 81,14 % de chances de rembourser. »

Pour employer cette formulation, il faudrait notamment :

- connaître précisément la signification métier de la cible ;
- évaluer le modèle sur des données indépendantes et représentatives ;
- calibrer ses probabilités ;
- vérifier la courbe de calibration et le score de Brier ;
- surveiller les biais et l’évolution des données dans le temps.

## Structure du projet

```text
TP_ML/
├── api.py
├── credit_scoring.csv
├── credit.data
├── credit_scoring.pkl
├── credit_scoring_cv.pkl
├── tp1.ipynb
├── utils.py
├── utils_part2.py
├── templates/
│   └── index.html
├── static/
│   ├── app.css
│   └── app.js
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Cloner le dépôt

```bash
git clone ADRESSE_DU_DEPOT
cd NOM_DU_DEPOT
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le modèle `credit_scoring.pkl` a été sérialisé avec **scikit-learn 1.7.2**. Cette version est donc fixée dans `requirements.txt` pour limiter les problèmes de compatibilité au chargement.

## Lancer l’application

Depuis le dossier contenant `api.py` :

```bash
python -m uvicorn api:app --reload
```

Adresses utiles :

| Service | Adresse |
| --- | --- |
| Interface web | `http://127.0.0.1:8000/app` |
| Documentation Swagger | `http://127.0.0.1:8000/docs` |
| État de l’API | `http://127.0.0.1:8000/health` |
| Racine de l’API | `http://127.0.0.1:8000/` |

## Tester l’API avec `curl`

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "Seniority": 9,
    "Home": 1,
    "Time": 60,
    "Age": 30,
    "Marital": 0,
    "Records": 1,
    "Job": 1,
    "Expenses": 73,
    "Income": 129,
    "Assets": 0,
    "Debt": 0,
    "Amount": 800,
    "Price": 846
  }'
```

Exemple de réponse :

```json
{
  "prediction": 0,
  "proba_classe_0": 0.9261208625905841,
  "proba_classe_1": 0.07387913740941582,
  "proba_classe_1_percent": 7.39
}
```

## Notebook et expérimentation

Pour ouvrir le notebook :

```bash
jupyter lab
```

Puis ouvrir :

```text
tp1.ipynb
```

Le notebook contient notamment :

- comparaison des classifieurs ;
- matrices de confusion ;
- accuracy, précision, rappel et AUC ;
- normalisation avec `StandardScaler` ;
- ajout de trois composantes PCA ;
- importance des variables ;
- sélection de variables ;
- analyse SHAP ;
- optimisation avec `GridSearchCV` ;
- validation croisée à 10 plis ;
- comparaison entre le scoring local et les réponses de l’API.

## Limites et usage responsable

Le credit scoring peut reproduire des biais présents dans les données historiques. Une utilisation réelle nécessiterait au minimum :

- une documentation précise des variables et des catégories ;
- une analyse d’équité entre différents groupes ;
- une calibration des probabilités ;
- une procédure de validation humaine ;
- une protection renforcée des données personnelles ;
- une surveillance des performances après déploiement.

Cette application doit être considérée comme une **démonstration universitaire de machine learning et de déploiement d’un modèle**.

## Améliorations envisagées

- Remplacer les codes des variables catégorielles par des libellés métier vérifiés.
- Calibrer les probabilités avec `CalibratedClassifierCV`.
- Ajouter une courbe de calibration et le score de Brier.
- Ajouter des tests automatiques avec `pytest`.
- Mettre en place une pipeline GitHub Actions.
- Conteneuriser l’application avec Docker.
- Ajouter une explication locale des prédictions.
- Déployer une démonstration publique sans données personnelles.

## Auteur

**Miloud Meziane** — projet universitaire de machine learning.
