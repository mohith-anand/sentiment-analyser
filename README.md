# Sentiment Analyser

A sentiment classification service — FastAPI backend, Streamlit frontend, reproducible training, CI-gated deployment, production monitoring.

**Live demo:** 
https://sentiment-analyser-wjzktgzblqbtappwwvey5t9.streamlit.app

![image alt](https://github.com/mohith-anand/sentiment-analyser/blob/9b1df49c181adda1001c06bc1233845894cad7d5/Screenshot%202025-10-15%20032630.png)

## How it was built

- **Dataset:** Kaggle Tweet Sentiment Extraction (~27k tweets, 3-class, balanced).
- **Preprocessing:** preserved negation words and sentiment punctuation (`not`, `isn't`, `!`, `?`) so phrases like "not good" retain their signal.
- **Models evaluated:** Logistic Regression, Linear SVC (calibrated), SGD — selected via 5-fold CV on macro-F1. Linear SVC won.
- **Added the MLOps layer piece by piece:** FastAPI splits inference from UI, tests catch regressions, CI retrains and blocks deploy if accuracy drops below 0.60.

## Architecture

```
Streamlit UI (Streamlit Cloud)  ──HTTPS──▶  FastAPI backend (Render)
                                 ◀──JSON───   /predict /health /stats
                                                 → SQLite prediction log
```

The frontend never loads the model — it calls the backend's `/predict` endpoint.

## Model

| Metric | Score |
|---:|---:|
| Test accuracy | 69.7% |
| Test macro-F1 | 0.699 |

Pipeline: TF-IDF (uni+bigrams, negation-aware) → Linear SVC (calibrated). See `model/metrics.txt` for the latest run.

**Known limitation:** the bag-of-words model struggles with sarcasm and certain pragmatic cues — confirmed via logged production predictions, not just offline metrics.

## MLOps pipeline (GitHub Actions)

Every push/PR to `main` runs, in order:

- **`test`** — installs `requirements-api.txt`, runs `pytest`.
- **`validate-model`** — reinstalls, retrains via `train.py`, fails if `test_accuracy` in `model/metrics.txt` drops below 0.60.
- **`build-and-push`** — builds and pushes the Docker image, only if both prior jobs pass.

A degraded model or a failing test can't reach production.

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","model_loaded":true}` |
| `POST` | `/predict` | `{"text": "..."}` → `{"label": "Positive\|Neutral\|Negative", "confidence": 0.0}` |
| `GET` | `/stats` | `total_count`, `count_per_label`, `recent_predictions` |

Security & behavior notes:
- The backend applies per-IP rate limiting to `/predict` (default `10/minute`) via `slowapi` to mitigate abuse.
- Inputs larger than 5000 characters are rejected with HTTP 400 to avoid resource exhaustion.
- `/stats` intentionally omits raw user-submitted text to protect privacy; only `id`, `predicted_label`, `confidence`, and `timestamp` are exposed.

## Deployment

**Backend (Render, Docker)** — the `Dockerfile` serves the app via `uvicorn api:app`, installing `requirements-api.txt`. Create a Web Service pointed at this repo; Render builds the image automatically. Set `ALLOWED_ORIGINS` (comma-separated) to the Streamlit app's URL in production — avoid `*`.

Implementation notes:
- `api.py` uses a FastAPI lifespan handler to load model artifacts and initialize the database at startup (replaces deprecated `@app.on_event("startup")`).
- Prediction logs are stored in a local `predictions.db` by default; we recommend replacing this with a managed data store in production (Supabase, Neon, or Postgres).
- To remove previously stored raw inputs from the local DB, run `python scrub_db.py` which creates a timestamped backup and then nulls or redacts the `text` column.

**Frontend (Streamlit Community Cloud)** — create an app pointed at this repo and `app.py`, then add the secret:
```
API_URL = "https://yourbackend.onrender.com"
```

## Run locally

```powershell
# install
python -m pip install -r requirements-api.txt -r requirements-app.txt

# (optional) retrain
python train.py

# backend
uvicorn api:app --reload --port 8000

# frontend, separate shell
$env:API_URL = "http://127.0.0.1:8000"
streamlit run app.py

# tests
python -m pytest -q

# docker smoke test
docker build -t sentiment-api .
docker run -e PORT=8000 -e ALLOWED_ORIGINS="http://localhost:8501" -p 8000:8000 sentiment-api
```

## Notes & security

- `predictions.db` is gitignored to avoid committing user logs. For production, use a managed DB (Supabase/Neon) — Render's free tier wipes local files on restart.
- Set `ALLOWED_ORIGINS` precisely in production rather than leaving CORS wide open.

## Roadmap

- `model/metadata.json` written during training (git SHA, timestamp, dataset hash, metrics) for full reproducibility
- Persistent prediction log with dashboards/alerting
- Golden-input regression tests to catch silent model drift
- Fine-tuned transformer baseline for comparison against the classical-ML ceiling