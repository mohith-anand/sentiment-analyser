import re
import string
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load vectorizer (make sure vectorizer.pkl is in the same folder)
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def transform_text(text):
    cleaned = clean_text(text)
    return vectorizer.transform([cleaned])
