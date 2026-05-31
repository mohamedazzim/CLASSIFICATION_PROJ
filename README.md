# Monorepo — Attrition ML + AutoDash Frontend

This repository contains a monorepo combining an Employee Attrition ML project (backend) and an automotive-themed frontend demo.

Structure

- `frontend/` — Static site (HTML/CSS/JS) with a dark automotive dashboard UI.
- `backend/app/models/` — Trained model file `model.pkl` used by the API.
- `api/predict.py` — FastAPI serverless endpoint used by Vercel and the frontend (`POST /api/predict`).
- `notebooks/` — Jupyter notebooks.
- `data/` — Dataset CSV files.

Quickstart (local)

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a local ASGI server for testing the API:

```bash
uvicorn api.predict:app --reload --port 8000
```

3. Open the frontend by serving `frontend/index.html` (or open directly in browser) and ensure `/api/predict` is reachable.

Deployment

This repository is configured for Vercel. Link the GitHub repo to Vercel and deploy — Vercel will run the Python serverless function under `/api/predict` and serve the static `frontend` directory.

Notes

- The frontend demo shows an automotive-themed dashboard but the backend model predicts employee attrition. The frontend sends a small JSON payload to `/api/predict` for demonstration — map fields appropriately for real use.
- The trained model is stored at `backend/app/models/model.pkl` and is included in the repository for demo purposes.
