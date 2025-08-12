# Fake News Moderator / Classifier (MVP)

A minimal end-to-end app to classify news as Fake/Real:
- Frontend: React (CDN) single-page app served statically
- Backend: FastAPI with a simple heuristic classifier and an optional trained stacked model

## Quick start (serve frontend + backend)

Requirements:
- Python 3.11
- pip and git

Install backend deps and start both servers:

1. Backend deps
   - cd fnmvp/fake-news-moderator-mvp/backend
   - pip install -r requirements.txt

2. Run both servers (from repo root)
   - PowerShell: PowerShell -ExecutionPolicy Bypass -File .\run_all.ps1
   - Backend: http://127.0.0.1:8000 (health: /healthz)
   - Frontend: http://localhost:5500/index.html

## Environment variables

Place secrets in backend/.env (gitignored). Example:

OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
SENTENCE_EMBEDDING_MODEL=all-MiniLM-L6-v2
SERPER_API_KEY=...
SERPER_BOOTSTRAP_QUERY=site:reuters.com OR site:apnews.com fact check

## Train a model (fast mode)

We provide a training script that can save model artifacts used by the API.

Prepare data (merges Fake.csv and True.csv into train/test):

- cd fnmvp/fake-news-moderator-mvp/backend
- python prepare_data.py

Run a quick training (fast mode, no NLI) and save artifacts:

- python stack4.py --fast --no_nli --train_csv data/train.csv --test_csv data/test.csv --output data/results_fast.csv

Artifacts will be saved to:

- fnmvp/fake-news-moderator-mvp/backend/models/
  - content/ (HF tokenizer+model)
  - context_model.joblib, context_scaler.joblib
  - meta_learner.joblib, config.json

Once artifacts exist, the API will automatically use them and /healthz will show:

- {"ok": true, "model_loaded": true}

## Full training (higher quality)

- python stack4.py --train_csv data/train.csv --test_csv data/test.csv --output data/results.csv

Notes:
- Full CPU runs can take hours; prefer GPU for the content model.
- You can enable the NLI branch by omitting --no_nli (heavier).

## Frontend config

The frontend points to API_BASE = http://127.0.0.1:8000 by default (see frontend/index.html).

## Development tips

- Backend hot reload uses uvicorn --reload
- CORS is open for local development
- Heuristic classifier remains as a fallback if artifacts are missing
