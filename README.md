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
