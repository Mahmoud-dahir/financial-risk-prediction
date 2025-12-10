import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# IMPORTANT : On doit importer la classe HybridACP pour que pickle la reconnaisse
from utils import HybridACP 

# 1. Initialisation de l'API
app = FastAPI(
    title="API Credit Scoring",
    description="API pour prédire l'octroi de crédit (MLP + ACP Hybride)",
    version="1.0"
)

# 2. Chargement du Pipeline entraîné
try:
    with open("credit_scoring.pkl", "rb") as f:
        pipeline = pickle.load(f)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    pipeline = None

# 3. Définition du format des données d'entrée (13 variables)
# L'ordre doit correspondre EXACTEMENT à ton X_train
class CreditClient(BaseModel):
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

# 4. Endpoint de Prédiction
@app.post("/predict/")
def predict_credit(client: CreditClient):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    # Conversion des données reçues en tableau numpy (1 ligne, 13 colonnes)
    # On respecte l'ordre des colonnes du CSV d'origine
    features = np.array([[
        client.Seniority, client.Home, client.Time, client.Age, 
        client.Marital, client.Records, client.Job, client.Expenses, 
        client.Income, client.Assets, client.Debt, client.Amount, client.Price
    ]])
    
    # Prédiction (Le pipeline gère tout : ACP, Sélection, MLP...)
    prediction = pipeline.predict(features)
    proba = pipeline.predict_proba(features)
    
    # Résultat
    resultat = "Accordé" if prediction[0] == 1 else "Refusé"
    confiance = proba[0][1] if prediction[0] == 1 else proba[0][0]
    
    return {
        "prediction_class": int(prediction[0]),
        "resultat": resultat,
        "probabilite_succes": round(proba[0][1], 4),
        "probabilite_echec": round(proba[0][0], 4),
        "message": f"Crédit {resultat} avec une confiance de {confiance:.1%}"
    }

# Endpoint de test simple
@app.get("/")
def read_root():
    return {"status": "L'API est en ligne et fonctionnelle 🚀"}
