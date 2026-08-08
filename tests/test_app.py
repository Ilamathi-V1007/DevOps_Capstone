import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "alcohol": 13.0,
    "malic_acid": 2.0,
    "ash": 2.3,
    "alcalinity_of_ash": 18.0,
    "magnesium": 100.0,
    "total_phenols": 2.5,
    "flavanoids": 2.5,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.5,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 3.0,
    "proline": 1000.0,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_valid_class():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in (0, 1, 2)
    assert body["class_probabilities"] is not None
    assert abs(sum(body["class_probabilities"]) - 1.0) < 1e-6


def test_predict_missing_field_returns_422():
    bad_payload = dict(SAMPLE_PAYLOAD)
    del bad_payload["alcohol"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
