import os
import sqlite3
import datetime
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from preprocess import clean_text

app = FastAPI(title="Sentiment Analyzer API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'sentiment_model.pkl')
vector_path = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

model = None
vectorizer = None

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    global model, vectorizer
    init_db()
    if not os.path.exists(model_path) or not os.path.exists(vector_path):
        raise RuntimeError("Model files not found! Please run training first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vector_path)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: Optional[float]

@app.get("/health")
def health():
    model_loaded = (model is not None) and (vectorizer is not None)
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace only")

    # Clean text and transform
    cleaned = clean_text(request.text)
    vect_text = vectorizer.transform([cleaned])
    
    # Predict
    pred_num = int(model.predict(vect_text)[0])
    label = label_map.get(pred_num, "Unknown")
    
    # Estimate confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vect_text)[0]
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if pred_num in classes:
                confidence = float(probs[classes.index(pred_num)])
            else:
                confidence = float(max(probs))
        else:
            if pred_num < len(probs):
                confidence = float(probs[pred_num])
            else:
                confidence = float(max(probs))

    # Log to SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO predictions (text, predicted_label, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        """, (request.text, label, confidence, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database logging failed: {e}")

    return PredictResponse(label=label, confidence=confidence)

@app.get("/stats")
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_count = cursor.fetchone()[0]
        
        # Count per label
        cursor.execute("SELECT predicted_label, COUNT(*) FROM predictions GROUP BY predicted_label")
        counts = cursor.fetchall()
        count_per_label = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for label, count in counts:
            if label in count_per_label:
                count_per_label[label] = count
                
        # Recent 5
        cursor.execute("SELECT id, text, predicted_label, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        recent_predictions = []
        for row in rows:
            recent_predictions.append({
                "id": row[0],
                "text": row[1],
                "predicted_label": row[2],
                "confidence": row[3],
                "timestamp": row[4]
            })
            
        conn.close()
        return {
            "total_count": total_count,
            "count_per_label": count_per_label,
            "recent_predictions": recent_predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query stats: {e}")
