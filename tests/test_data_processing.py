from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/heart_disease.csv")


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_shape():
    df = pd.read_csv(DATA_PATH)

    assert df.shape[0] == 303
    assert df.shape[1] == 14


def test_target_is_binary():
    df = pd.read_csv(DATA_PATH)

    df["target"] = (df["target"] > 0).astype(int)

    assert set(df["target"].unique()) == {0, 1}


def test_required_columns_exist():
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    }

    assert required_columns.issubset(df.columns)