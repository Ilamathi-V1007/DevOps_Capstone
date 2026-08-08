"""
Prediction utilities: loads the best model from the MLflow Model
Registry and the fitted scaler, and exposes a predict() function.
"""

import os
import joblib
import numpy as np
import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "wine-classifier"
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")

FEATURE_ORDER = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]

_model = None
_scaler = None


def _get_latest_model_uri():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for model '{REGISTERED_MODEL_NAME}'. "
            "Run `python src/train.py` first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{REGISTERED_MODEL_NAME}/{latest.version}", latest.version


def load_artifacts():
    global _model, _scaler
    if _model is None:
        model_uri, version = _get_latest_model_uri()
        _model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded {REGISTERED_MODEL_NAME} version {version}")
    if _scaler is None:
        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
        else:
            _scaler = None  # fall back to raw features if no scaler was saved
    return _model, _scaler


def predict(features: dict):
    model, scaler = load_artifacts()
    row = np.array([[features[f] for f in FEATURE_ORDER]])
    if scaler is not None:
        row = scaler.transform(row)
    pred = model.predict(row)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0].tolist()
    return int(pred), proba
