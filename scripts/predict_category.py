import os
import joblib
import numpy as np
from scipy.sparse import hstack
from scripts.data_processing import clean_text
from scripts.features import add_text_features
import pandas as pd

print("\n Product Category Predictor")

# === Setare căi absolute ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# === Încărcare model și vectorizator ===
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# === Buclă interactivă ===
while True:
    print()
    title = input("Titlu produs (sau 'exit' pentru a ieși): ").strip()
    if title.lower() == "exit":
        print("👋 Ieșire din aplicație.")
        break

    # === Preprocesare text ===
    clean_title = clean_text(title)

    # === Creare DataFrame pentru features suplimentare ===
    df = pd.DataFrame({"Product Title": [title]})
    df = add_text_features(df, "Product Title")
    X_features = df[["title_length", "word_count", "contains_number", "contains_upper"]].astype(float)

    # === Vectorizare text ===
    X_text = vectorizer.transform([clean_title])

    # === Combinare text + features ===
    X = hstack([X_text, X_features.values])

    # === Predicție ===
    pred = model.predict(X)[0]
    print(f"\n Predicția modelului: {pred}")
