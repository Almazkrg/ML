# ML FastAPI Docker

Iris classification (RandomForest) served via FastAPI and Docker.

## Run locally
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python train.py
    uvicorn main:app --reload

Swagger UI: http://127.0.0.1:8000/docs

## Run with Docker
    docker build -t ml-fastapi .
    docker run -p 8000:8000 ml-fastapi

## Endpoints
- GET / : health message
- POST /predict : returns the predicted iris class

Example body:
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

## MLflow (experiment tracking & model registry)
    python train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Open http://127.0.0.1:5000 (experiment `iris-classification`, model `iris-rf-model`).

## Streamlit frontend
    uvicorn main:app --reload
    streamlit run app.py

## Everything with Docker Compose
    docker compose build
    docker compose run --rm train
    docker compose up -d

- API and home page: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
