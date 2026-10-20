from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/heart_disease_logistic_regression.pkl")
DATA_PATH = Path("data/raw/heart_disease.csv")


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_model_predictions_are_binary():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)

    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})