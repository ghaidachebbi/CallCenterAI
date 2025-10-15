# src/api/tfidf_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Télécharger stopwords la première fois
nltk.download('stopwords')

# === Initialisation FastAPI ===
app = FastAPI(title="TF-IDF SVM Service")

# === Charger le modèle ===
model = joblib.load("models/tfidf_svm.pkl")

# === Définition des entrées ===
class Ticket(BaseModel):
    text: str

# === Fonction de nettoyage du texte ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Endpoint de prédiction ===
@app.post("/predict")
def predict(ticket: Ticket):
    # Nettoyage
    cleaned_text = clean_text(ticket.text)
    
    # Prédiction label
    pred = model.predict([cleaned_text])[0]
    
    # Probabilités
    try:
        probas = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probas))
    except AttributeError:
        # Si le modèle n'est pas calibré
        confidence = 1.0

    return {"label": pred, "confidence": confidence}

# === Endpoint pour Prometheus ou monitoring simple ===
@app.get("/metrics")
def metrics():
    return {"requests_total": 0, "requests_failed": 0}
