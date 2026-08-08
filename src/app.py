"""
FastAPI prediction service for the wine classification model.

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

Then POST to /predict, or open /docs for the Swagger UI.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict, load_artifacts, FEATURE_ORDER


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load the model/scaler so the first real request isn't slow
    # and so failures surface immediately at container start.
    load_artifacts()
    yield


app = FastAPI(
    title="Wine Classification API",
    description="Serves predictions from the MLflow-registered wine-classifier model.",
    version="1.0.0",
    lifespan=lifespan,
)


class WineFeatures(BaseModel):
    alcohol: float = Field(..., example=13.0)
    malic_acid: float = Field(..., example=2.0)
    ash: float = Field(..., example=2.3)
    alcalinity_of_ash: float = Field(..., example=18.0)
    magnesium: float = Field(..., example=100.0)
    total_phenols: float = Field(..., example=2.5)
    flavanoids: float = Field(..., example=2.5)
    nonflavanoid_phenols: float = Field(..., example=0.3)
    proanthocyanins: float = Field(..., example=1.5)
    color_intensity: float = Field(..., example=5.0)
    hue: float = Field(..., example=1.0)
    od280_od315_of_diluted_wines: float = Field(..., alias="od280/od315_of_diluted_wines", example=3.0)
    proline: float = Field(..., example=1000.0)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_class: int
    class_probabilities: list[float] | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Wine Classification API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: WineFeatures):
    try:
        payload = features.model_dump(by_alias=True)
        ordered = {f: payload[f] for f in FEATURE_ORDER}
        pred_class, proba = predict(ordered)
        return PredictionResponse(predicted_class=pred_class, class_probabilities=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
