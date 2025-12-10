
import requests
import json

# L'URL de votre API
url = 'http://127.0.0.1:8000/predict/'

# --- PROFIL RISQUÉ (Test de refus) ---
donnees_a_envoyer = {
    "Seniority": 0.0,
    "Home": 0.0,
    "Time": 36.0,
    "Age": 20.0,
    "Marital": 0.0,
    "Records": 0.0,
    "Job": 0.0,
    "Expenses": 50.0,
    "Income": 20.0,
    "Assets": 0.0,
    "Debt": 3000.0,
    "Amount": 1500.0,
    "Price": 1600.0
}

try:
    print("--- Envoi du profil RISQUÉ à l'API ---")
    response = requests.post(url, json=donnees_a_envoyer)

    if response.status_code == 200:
        resultat = response.json()
        print("\n✅ Réponse reçue :")
        print(f"Décision : {resultat.get('resultat', 'Inconnue')}")
        print(f"Message  : {resultat.get('message', '')}")
        # On gère le cas où la clé serait différente
        prob = resultat.get('probabilite_echec') or resultat.get('probabilite_refus', 0)
        print(f"Probabilité d'échec : {prob*100:.2f}%")
    else:
        print(f"\n❌ Erreur {response.status_code} : {response.text}")

except requests.exceptions.ConnectionError:
    print("\n❌ Impossible de se connecter. Vérifiez que le serveur (app.py) tourne dans l'autre terminal.")