import joblib
from data_processing import clean_text
import numpy as np

# Încarcă modelul și vectorizatorul
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("🔮 Product Category Predictor")
print("Introduceți titlul produsului (sau 'exit' pentru a ieși):")

while True:
    title = input("\nTitlu produs: ")
    if title.lower() == "exit":
        break
    clean = clean_text(title)
    X = vectorizer.transform([clean])
    pred = model.predict(X)
    print(f"➡️ Categoria prezisă: {pred[0]}")
