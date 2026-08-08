FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, registered model store, and data
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY mlflow.db ./mlflow.db
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Startup configuration: run the FastAPI app with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
