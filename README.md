
##  Instalare și configurare

1. Clonează proiectul:
   ```bash
   git clone https://github.com/razvan-stegaru/product-category-classifier.git
   cd product-category-classifier

## Structura proiectului
product-category-classifier/
│
├── data/ # Seturi de date pentru antrenare/testare
│ └── products.csv
│
├── models/ # Modele salvate
│ ├── model.pkl # Modelul Logistic Regression antrenat
│ └── vectorizer.pkl # TF-IDF vectorizer folosit pentru text
│
├── scripts/ # Scripturi principale
│ ├── train_model.py # Script pentru antrenarea modelului
│ ├── predict_category.py # Script pentru prezicerea categoriei
│ ├── data_processing.py # Curățarea și preprocesarea textului
│ └── features.py # Generarea de features numerice suplimentare
│
└── README.md

## Cerinte 
- Python **3.9+** (recomandat 3.10–3.12)
- pip install -r requirements.txt
## Exemplu de rulare

> python -m scripts.train_model
 Modelul a fost antrenat și salvat în /models

> python -m scripts.predict_category
 Product Category Predictor
Titlu produs: Samsung Galaxy A52 128GB

> Predicția modelului: Mobile Phones