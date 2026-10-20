import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

MODEL_PATH = Path("models/heart_disease_logistic_regression.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heart Disease Prediction API")

# API request logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logger.info(
        "%s %s - Status: %s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


# Prometheus metrics
Instrumentator().instrument(app).expose(app)


class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "confidence": confidence,
    }