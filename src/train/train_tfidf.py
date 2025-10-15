# train_tfidf.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import mlflow

# Télécharger les stopwords la première fois
nltk.download('stopwords')

# === 1️⃣ Fonction de nettoyage du texte ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Minuscule
    text = text.lower()
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    # Supprimer les mots vides (stopwords) mais garder certains mots clés
    stop_words = set(stopwords.words('english'))
    keywords_keep = {"invoice", "payment", "subscription", "charge", "billing"}
    text = ' '.join([word for word in text.split() if word not in stop_words or word in keywords_keep])
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === 2️⃣ Charger le dataset ===
df = pd.read_csv("data/tickets.csv")

# Vérification des colonnes
if 'Document' not in df.columns or 'Topic_group' not in df.columns:
    raise ValueError("❌ Les colonnes 'Document' et 'Topic_group' doivent exister dans data/tickets.csv")

# Nettoyage du texte
df['clean_text'] = df['Document'].apply(clean_text)

texts = df['clean_text'].tolist()
labels = df['Topic_group'].tolist()

# === 3️⃣ Division train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# === 4️⃣ Pipeline TF-IDF + SVM calibré ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,3))),
    ('svc', CalibratedClassifierCV(LinearSVC()))
])

# === 5️⃣ Entraînement et suivi avec MLflow ===
mlflow.start_run()

pipeline.fit(X_train, y_train)

# === 6️⃣ Évaluation ===
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print("📊 Rapport de classification :")
print(report)

# Log des métriques principales dans MLflow
mlflow.log_metrics({
    "accuracy": report["accuracy"],
    "f1_macro": report["macro avg"]["f1-score"]
})

mlflow.end_run()

# === 7️⃣ Sauvegarde du modèle ===
joblib.dump(pipeline, "models/tfidf_svm.pkl")

print("\n✅ Modèle entraîné et sauvegardé dans models/tfidf_svm.pkl")
