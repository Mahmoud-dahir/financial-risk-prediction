# api.py
from pathlib import Path
import pickle

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(
    title="Credit Scoring API",
    description="API FastAPI pour scorer un individu à partir du pipeline sauvegardé.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_scoring.pkl"

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)
pipeline = None  # chargé au démarrage (startup)


class CreditRequest(BaseModel):
    """
    Schéma attendu par l'API : 13 variables numériques (mêmes noms que le CSV).
    """
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


@app.on_event("startup")
def load_model():
    """
    Chargement du pipeline au lancement de l'API.
    Ça évite de relire le pickle à chaque requête.
    """
    global pipeline

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fichier '{MODEL_PATH}' introuvable. "
            f"Vérifie que tu as bien généré le pipeline et que tu es dans le bon dossier."
        )

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    print("Pipeline chargé depuis 'credit_scoring.pkl'")


@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de credit scoring",
        "usage": "Appelle POST /predict/ avec les 13 features.",
    }

@app.get("/app", response_class=HTMLResponse)
def credit_scoring_app(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

@app.get("/health")
def health():
    """
    Endpoint simple pour vérifier que le modèle est bien chargé.
    """
    return {"model_loaded": pipeline is not None}


@app.post("/predict/")
def predict_credit(req: CreditRequest):
    """
    Prend un individu en entrée et renvoie :
      - la classe prédite (0/1)
      - les probabilités (si disponibles) + un pourcentage lisible
    """
    # Ordre identique au CSV (colonnes features)
    x = np.array(
        [[
            req.Seniority, req.Home, req.Time, req.Age, req.Marital,
            req.Records, req.Job, req.Expenses, req.Income, req.Assets,
            req.Debt, req.Amount, req.Price,
        ]],
        dtype=float,
    )

    pred = int(pipeline.predict(x)[0])

    proba_0, proba_1 = None, None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(x)[0]
        proba_0 = float(proba[0])
        proba_1 = float(proba[1])

    return {
        "prediction": pred,
        "proba_classe_0": proba_0,
        "proba_classe_1": proba_1,
        "proba_classe_1_percent": None if proba_1 is None else round(proba_1 * 100, 2),
    }
