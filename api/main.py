from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI

from api.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_random_forest.joblib"
)


model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Fraud Detection API",
    description="Machine Learning API for credit card fraud detection",
    version="1.0.0",
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "random_forest",
    }


@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
)
def predict(
    request: FraudPredictionRequest,
):

    data = pd.DataFrame(
        [request.model_dump()]
    )

    probability = model.predict_proba(data)[0][1]

    prediction = int(
        model.predict(data)[0]
    )

    return {
        "fraud_probability": float(probability),
        "prediction": prediction,
    }