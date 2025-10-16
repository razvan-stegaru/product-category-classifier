import re
import nltk
from nltk.corpus import stopwords

# Descărcăm stopwords la prima rulare
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)
