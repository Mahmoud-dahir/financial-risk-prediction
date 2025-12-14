# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path

# =====================================================================
# 0. INITIALISATION FASTAPI
# =====================================================================

app = FastAPI(
    title="Credit Scoring API",
    description="API FastAPI pour scorer un individu à partir du pipeline sauvegardé.",
    version="1.0.0"
)

# =====================================================================
# 1. CHARGEMENT DU PIPELINE PICKLE
# =====================================================================

MODEL_PATH = Path("credit_scoring.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fichier '{MODEL_PATH}' introuvable. "
        f"Vérifie que tu as bien lancé pipeline_generation_train_test_split "
        f"et que tu es dans le bon dossier."
    )

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

print("✅ Pipeline chargé depuis 'credit_scoring.pkl'")


# =====================================================================
# 2. SCHÉMA D'ENTRÉE (Pydantic)
# =====================================================================

class CreditRequest(BaseModel):
    Seniority: float
    Home: float
    Time: float
    Age: float
    Marital: float
    Records: float
    Job: float
    Expenses: float
    Income: float
    Assets: float
    Debt: float
    Amount: float
    Price: float


# =====================================================================
# 3. ENDPOINTS
# =====================================================================

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de credit scoring 👋",
        "usage": "Appelle le endpoint POST /predict/ avec les 13 features."
    }


@app.post("/predict/")
def predict_credit(req: CreditRequest):
    """
    Reçoit un individu (13 features) et renvoie :
      - la prédiction (0 ou 1)
      - la probabilité brute de la classe 1
      - la probabilité de la classe 1 en %
    """
    # On construit le vecteur X dans le BON ordre :
    x = np.array([[
        req.Seniority,
        req.Home,
        req.Time,
        req.Age,
        req.Marital,
        req.Records,
        req.Job,
        req.Expenses,
        req.Income,
        req.Assets,
        req.Debt,
        req.Amount,
        req.Price
    ]], dtype=float)

    # Probabilités et prédiction
    proba = pipeline.predict_proba(x)[0]  # [proba_classe_0, proba_classe_1]
    proba_0 = float(proba[0])
    proba_1 = float(proba[1])
    pred = int(pipeline.predict(x)[0])

    return {
        "prediction": pred,
        "proba_classe_0": proba_0,
        "proba_classe_1": proba_1,
        "proba_classe_1_percent": round(proba_1 * 100, 2)  # ex: 22.99
        # Si tu veux aussi une version string :
        # "proba_classe_1_str": f"{proba_1 * 100:.2f}%"
    }
