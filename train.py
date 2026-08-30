

import re
import string
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RANDOM_STATE = 42
DATA_PATH = "train.csv"
OUT_DIR = Path("model")

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Negation words matter a lot for sentiment — never strip these even though
# they're in generic English stopword lists. This alone fixes a common bug
# in naive TF-IDF pipelines (e.g. "not good" losing its "not").
NEGATIONS = {
    "no", "not", "nor", "none", "never", "n't", "cannot", "can't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    "but", "however", "although",
}
STOPWORDS = sk_text.ENGLISH_STOP_WORDS - NEGATIONS


def clean_text(text: str) -> str:
    """Light tweet cleaning that preserves negation and punctuation cues
    (like '!' and '?') that carry sentiment signal."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                     # keep hashtag text, drop '#'
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                # sooo -> soo (de-elongate)
    # keep letters, spaces, apostrophes (contractions), and ! / ? (sentiment signal)
    text = re.sub(r"[^a-z\s'!?]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df = df[df["sentiment"].isin(LABEL_MAP)]
    df["label"] = df["sentiment"].map(LABEL_MAP)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


def build_candidates():
    """A few sklearn-compatible models worth comparing. All are fast enough
    to cross-validate on ~27k rows in well under a minute."""
    return {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=5, random_state=RANDOM_STATE
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, random_state=RANDOM_STATE),
            cv=3,
        ),
        "sgd": SGDClassifier(
            loss="modified_huber", class_weight="balanced",
            alpha=1e-5, random_state=RANDOM_STATE,
        ),
    }


def main():
    print("Loading and cleaning data...")
    df = load_data()
    print(f"  {len(df)} usable rows after cleaning")
    print(f"  class balance:\n{df['sentiment'].value_counts()}\n")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=40000,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Comparing candidate models (5-fold CV, macro-F1)...")
    best_name, best_model, best_cv = None, None, -1
    for name, model in build_candidates().items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        print(f"  {name:12s} macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_cv:
            best_name, best_model, best_cv = name, model, scores.mean()

    print(f"\nBest model: {best_name} (CV macro-F1 = {best_cv:.4f})")
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    acc = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nHeld-out test set:")
    print(f"  accuracy  = {acc:.4f}")
    print(f"  macro-F1  = {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))
    print("Confusion matrix (rows=true, cols=pred; order neg/neu/pos):")
    print(confusion_matrix(y_test, y_pred))

    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, OUT_DIR / "sentiment_model.pkl")
    joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.pkl")
    print(f"\nSaved {OUT_DIR / 'sentiment_model.pkl'} and {OUT_DIR / 'tfidf_vectorizer.pkl'}")

    # Save a small metrics report alongside the model — useful for your
    # README / resume claim, and for CI to check against later.
    with open(OUT_DIR / "metrics.txt", "w") as f:
        f.write(f"model: {best_name}\n")
        f.write(f"cv_macro_f1: {best_cv:.4f}\n")
        f.write(f"test_accuracy: {acc:.4f}\n")
        f.write(f"test_macro_f1: {macro_f1:.4f}\n")


if __name__ == "__main__":
    main()
