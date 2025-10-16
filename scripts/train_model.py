import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import hstack
from scripts.data_processing import clean_text
from scripts.features import add_text_features

# === 1. Încărcare set de date ===
df = pd.read_csv(r"D:\LINKACADEMY\ML\tema3\product-category-classifier\data\products.csv")

# elimină spațiile din numele coloanelor
df.columns = df.columns.str.strip()

# === 2. Eliminare valori lipsă ===
# dacă lipsesc coloane, înlocuiește cu numele corect din CSV-ul tău (ex. "Category")
if "Product Title" not in df.columns or "Category Label" not in df.columns:
    print(" Verifică numele coloanelor din CSV (ex: 'Product Title' și 'Category Label')")
    print("Coloane găsite:", list(df.columns))
    exit()

# elimină rândurile fără titlu sau categorie
df = df.dropna(subset=["Product Title", "Category Label"])

# === 3. Curățare text ===
df["clean_title"] = df["Product Title"].apply(clean_text)

# === 4. Adăugare caracteristici suplimentare ===
df = add_text_features(df, "Product Title")

# === 5. Vectorizare text (TF-IDF) ===
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["clean_title"])

# === 6. Combinare text + features numerice ===
X_features = df[["title_length", "word_count", "contains_number", "contains_upper"]].astype(float)
X = hstack([X_text, X_features.values])
y = df["Category Label"]

# === 7. Împărțire în train/test ===
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print(" Atenție: Unele categorii au prea puține exemple pentru stratificare.")
    print("Se va folosi împărțire simplă fără stratify.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# === 8. Antrenare model ===
print("\n Se antrenează modelul LogisticRegression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 9. Evaluare model ===
y_pred = model.predict(X_test)

print("\n Acuratețe:", round(accuracy_score(y_test, y_pred), 4))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# === 10. Matrice de confuzie ===
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# === 11. Salvare model și vectorizator ===
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n Modelul și vectorizatorul au fost salvate în folderul 'models/'.")
