"""import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialisation de l'API
app = FastAPI(title="API Crédit Scoring", description="Prédiction de l'octroi de crédit")

# 2. Chargement du modèle et des indices de colonnes
# On charge le pipeline (Cerveau)
with open('credit_scoring.pkl', 'rb') as f:
    model = pickle.load(f)

# On charge la liste des variables utiles (Filtre)
with open('features_indices.pkl', 'rb') as f:
    features_indices = pickle.load(f)

# 3. Définition du format des données d'entrée (Basé sur ton CSV)
class CreditInput(BaseModel):
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

# 4. Création de l'endpoint (la "porte d'entrée")
@app.post("/predict/")
def predict_credit(data: CreditInput):
    # a. Conversion des données reçues en liste
    # L'ordre doit être STRICTEMENT le même que dans le CSV
    data_list = [
        data.Seniority, data.Home, data.Time, data.Age, data.Marital, 
        data.Records, data.Job, data.Expenses, data.Income, data.Assets, 
        data.Debt, data.Amount, data.Price
    ]
    
    # b. Transformation en tableau Numpy (1 ligne, 13 colonnes)
    data_array = np.array(data_list).reshape(1, -1)
    
    # c. Sélection des meilleures variables (Étape cruciale !)
    # Le modèle a été entraîné avec seulement 8 variables, il faut lui donner les mêmes.
    data_optimized = data_array[:, features_indices]
    
    # d. Prédiction
    prediction = model.predict(data_optimized)
    proba = model.predict_proba(data_optimized)
    
    # e. Renvoi de la réponse (JSON)
    # prediction[0] vaut 1 (Accordé) ou 0 (Refusé)
    result = "Accordé" if prediction[0] == 1 else "Refusé"
    
    return {
        "prediction": result,
        "probabilité_refus": round(proba[0][0], 2),
        "probabilité_accord": round(proba[0][1], 2)
    }
"""
