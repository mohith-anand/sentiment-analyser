import os
import sys
import subprocess
import joblib
import pytest

from preprocess import clean_text


def ensure_models():
    """Run training to produce model artifacts if they don't exist."""
    model_file = os.path.join("model", "sentiment_model.pkl")
    vec_file = os.path.join("model", "tfidf_vectorizer.pkl")
    if not (os.path.exists(model_file) and os.path.exists(vec_file)):
        subprocess.check_call([sys.executable, "train.py"])


def test_clean_text_empty_and_url_and_negation():
    assert clean_text("") == ""

    s = "Check this out http://example.com and www.example.com/page"
    cleaned = clean_text(s)
    assert "http" not in cleaned and "www" not in cleaned and "example" not in cleaned

    neg = "not good"
    neg_clean = clean_text(neg)
    assert "not" in neg_clean


def test_vectorizer_transform_and_model_predict():
    ensure_models()
    vec = joblib.load(os.path.join("model", "tfidf_vectorizer.pkl"))
    X = vec.transform(["this is a sample sentence for testing"])
    # Ensure we got a non-empty sparse matrix
    assert hasattr(X, "nnz") and X.nnz > 0

    model = joblib.load(os.path.join("model", "sentiment_model.pkl"))
    pred = model.predict(X)[0]
    try:
        pred_int = int(pred)
    except Exception:
        pred_int = None
    assert pred in {0, 1, 2} or pred_int in {0, 1, 2}


def test_api_endpoints():
    ensure_models()
    from api import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200

    r = client.post("/predict", json={"text": "I love this"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("label") in {"Positive", "Negative", "Neutral"}

    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 400
