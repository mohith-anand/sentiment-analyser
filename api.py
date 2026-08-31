import os
import sqlite3
import datetime
import joblib
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from preprocess import clean_text
from fastapi.middleware.cors import CORSMiddleware

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    # Ensure DB exists and models are available during startup
    init_db()
    if not os.path.exists(model_path) or not os.path.exists(vector_path):
        raise RuntimeError("Model files not found! Please run training first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vector_path)
    yield

# Create app with lifespan handler (replaces deprecated startup event)
app = FastAPI(title="Sentiment Analyzer API", lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'sentiment_model.pkl')
vector_path = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

model = None
vectorizer = None

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Configure CORS origins via ALLOWED_ORIGINS env var (comma-separated).
# Defaults to allow all origins in development; set to the Streamlit URL in production.
allowed = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed.strip() == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configure rate limiter (per-IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")

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

# startup handled by lifespan

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
@limiter.limit("10/minute")
def predict(request: Request, payload: PredictRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace only")

    # Reject overly long inputs to avoid resource exhaustion
    if len(payload.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long")

    # Clean text and transform
    cleaned = clean_text(payload.text)
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
        """, (payload.text, label, confidence, timestamp))
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
                
        # Recent 5 — omit raw text to avoid exposing user-submitted content
        cursor.execute("SELECT id, predicted_label, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        recent_predictions = []
        for row in rows:
            recent_predictions.append({
                "id": row[0],
                "predicted_label": row[1],
                "confidence": row[2],
                "timestamp": row[3]
            })
            
        conn.close()
        return {
            "total_count": total_count,
            "count_per_label": count_per_label,
            "recent_predictions": recent_predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query stats: {e}")
