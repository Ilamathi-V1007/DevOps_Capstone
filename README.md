# MLOps Capstone: Wine Classification Pipeline

An end-to-end MLOps pipeline: DVC-versioned data → 3-model training with MLflow
tracking → best model registered in the MLflow Model Registry → FastAPI
prediction service → Docker container → GitHub Actions CI/CD.

**Dataset:** Wine classification dataset (178 samples, 13 features, 3 classes),
loaded from scikit-learn's built-in `load_wine` and saved to `data/wine.csv`.

## Project Structure

```
project/
├── data/                  # DVC-tracked dataset
├── models/                # Saved preprocessing scaler (model itself lives in MLflow registry)
├── src/
│   ├── train.py           # Data loading, preprocessing, 3-model training, MLflow tracking
│   ├── app.py              # FastAPI app exposing POST /predict
│   ├── predict.py          # Loads registered model + scaler, runs inference
│   └── utils.py
├── tests/                 # pytest suite for the API and training pipeline
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Data Versioning (DVC)

```bash
dvc init
dvc add data/wine.csv
git add data/wine.csv.dvc data/.gitignore
dvc remote add -d localstorage <path-or-cloud-remote>
dvc push
```

## 2. Train models + track with MLflow

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python src/train.py
```

This trains Logistic Regression, Random Forest, and Gradient Boosting,
logs params/metrics/artifacts for each run, and registers the
best-performing model (by macro F1) as `wine-classifier` in the MLflow
Model Registry.

View experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 3. Run the prediction API

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI, or:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "alcohol": 13.0, "malic_acid": 2.0, "ash": 2.3, "alcalinity_of_ash": 18.0,
  "magnesium": 100.0, "total_phenols": 2.5, "flavanoids": 2.5,
  "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.5, "color_intensity": 5.0,
  "hue": 1.0, "od280/od315_of_diluted_wines": 3.0, "proline": 1000.0
}'
```

## 4. Run tests

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
pytest tests/ -v
```

## 5. Docker

```bash
docker build -t wine-classifier-api .
docker run -p 8000:8000 wine-classifier-api
```

## 6. CI/CD

`.github/workflows/ci.yml` runs on every push: checks out the repo, installs
dependencies, retrains the model (populating the registry + scaler), runs the
test suite, and builds the Docker image.

## Notes

- The MLflow tracking store here uses local SQLite (`sqlite:///mlflow.db`) so
  the Model Registry works without an external server. For a shared/team
  setup, point `MLFLOW_TRACKING_URI` at a hosted MLflow server instead.
- The DVC remote in this repo is set up locally for demonstration; swap it
  for a real remote (Google Drive, S3, etc.) before final submission if you
  want `dvc push`/`pull` to work from a fresh clone.
