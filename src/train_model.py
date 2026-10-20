# 2.1 Import Libraries, Load Dataset and convert target to binary

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/heart_disease.csv")
MODEL_PATH = Path("models/heart_disease_logistic_regression.pkl")

df = pd.read_csv(DATA_PATH)

# Convert target to binary
df["target"] = (df["target"] > 0).astype(int)


# 2.2 Define Features & Target, Split Train/Test

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 2.3 Define Numerical and Categorical Features

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


# 2.4 Preprocessing Pipeline

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 3.1 Logistic Regression

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=1,
        max_iter=1000,
        random_state=42
    ))
])

logistic_pipeline.fit(X_train, y_train)


# Save final model

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    logistic_pipeline,
    MODEL_PATH
)

print("Model training completed successfully.")
print("Model saved to:", MODEL_PATH)