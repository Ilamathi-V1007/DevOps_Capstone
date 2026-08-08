import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import load_data, preprocess, get_candidate_models, evaluate
from sklearn.model_selection import train_test_split


def test_load_data_shapes():
    X, y = load_data()
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    assert "target" not in X.columns


def test_preprocess_scales_features():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert scaler is not None


def test_three_candidate_models_defined():
    models = get_candidate_models()
    assert len(models) >= 3


def test_evaluate_returns_expected_metrics():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, _ = preprocess(X_train, X_test)
    model, _ = get_candidate_models()["logistic_regression"]
    model.fit(X_train_scaled, y_train)
    metrics = evaluate(model, X_test_scaled, y_test)
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
