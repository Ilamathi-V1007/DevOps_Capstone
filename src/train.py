"""
Training pipeline for the Wine Classification capstone project.

Loads data, preprocesses it, trains three candidate models, logs every
run (params, metrics, model artifact) to MLflow, and registers the
best-performing model in the MLflow Model Registry.

Usage:
    python src/train.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wine.csv")
EXPERIMENT_NAME = "wine-classification"
REGISTERED_MODEL_NAME = "wine-classifier"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
    }


def get_candidate_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, C=1.0),
            {"max_iter": 1000, "C": 1.0},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            {"n_estimators": 200, "max_depth": 6, "random_state": 42},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
        ),
    }


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    best_run_id = None
    best_model_name = None
    best_f1 = -1.0

    for model_name, (model, params) in get_candidate_models().items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate(model, X_test_scaled, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{model_name}] {metrics}")

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_run_id = run.info.run_id
                best_model_name = model_name

    print(f"\nBest model: {best_model_name} (run_id={best_run_id}, f1_macro={best_f1:.4f})")

    # Register the best model in the MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {registered.version}")


if __name__ == "__main__":
    main()
