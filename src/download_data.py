from pathlib import Path

import pandas as pd

# Create the raw data directory
RAW_DATA_DIR = Path("../data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# UCI Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names from the UCI dataset
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Download directly from UCI
df = pd.read_csv(
    url,
    header=None,
    names=columns,
    na_values="?"
)

# Save a local copy for reproducibility
output_path = RAW_DATA_DIR / "heart_disease.csv"
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print("Saved to:", output_path)
print("Shape:", df.shape)

print(df.head())