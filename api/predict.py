from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title='Attrition Prediction API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL_PATH = Path(__file__).resolve().parents[1] / 'backend' / 'app' / 'models' / 'model.pkl'
if not MODEL_PATH.exists():
    # fallback to common location
    alt = Path.cwd() / 'backend' / 'app' / 'models' / 'model.pkl'
    MODEL_PATH = alt

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print('Warning: model could not be loaded at startup:', e)

class PredictRequest(BaseModel):
    data: Dict[str, Any]

class PredictResponse(BaseModel):
    prediction: Any
    probability: float

@app.post('/api/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    try:
        # Build DataFrame with single row
        row = pd.DataFrame([req.data])
        # Ensure columns order matches training if possible
        # Model is a Pipeline expecting the original feature columns
        # We'll attempt to pass the row as-is and let the pipeline handle missing values
        prob = None
        pred = None
        try:
            prob = model.predict_proba(row)[:, 1][0]
            pred = model.predict(row)[0]
        except Exception:
            # Some models may not implement predict_proba
            pred = model.predict(row)[0]
            prob = float('nan')
        return PredictResponse(prediction=pred, probability=float(prob))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction failed: {e}')
