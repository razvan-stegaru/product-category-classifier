import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_proccesing import clean_text
from features import add_text_features

# 1. Încărcare set de date
df = pd.read_csv(r"D:\LINKACADEMY\ML\tema3\product-category-classifier\data\products.csv")

# 2. Curățare text
df["clean_title"] = df["Product Title"].apply(clean_text)

# 3. Adăugare caracteristici suplimentare
df = add_text_features(df, "Product Title")

# 4. Vectorizare text (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["clean_title"])

# 5. Combinare text + features numerice
X_features = df[["title_length", "word_count", "contains_number", "contains_upper"]].astype(float)
import numpy as np
from scipy.sparse import hstack
X = hstack([X_text, X_features.values])

df.columns = df.columns.str.strip()  # elimină spațiile la început și sfârșit
y = df["Category Label"]

# 6. Împărțire train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Antrenare model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluare model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Matrice de confuzie
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 10. Salvare model și vectorizator
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Modelul și vectorizatorul au fost salvate în folderul 'models/'")
