"""Fastapi server - main entry point. ONLY FOR deployment (model prediction), NOT training."""

import time

import mlflow
from fastapi import FastAPI, Request
from pyspark.sql import SparkSession

from train_pyspark_baseline import predict_single

app = FastAPI()

spark = SparkSession.builder.appName("prediction_server").getOrCreate()

MODEL_URI = "data/output/"
model = None


def get_model():
    global model
    if model is None:
        from pyspark.ml import PipelineModel

        model = PipelineModel.load(MODEL_URI)
    return model


@app.get("/")
def hello():
    return "Hello World"


@app.post("/predict")
def predict(current: dict, previous: dict):
    mlflow.set_experiment("nifty-stock-prediction")
    with mlflow.start_run(run_name="prediction_request"):
        start = time.time()

        loaded_model = get_model()
        prediction = predict_single(loaded_model, previous, current, spark)

        latency_ms = (time.time() - start) * 1000

        mlflow.log_metrics({"prediction_latency_ms": latency_ms})
        mlflow.log_params(
            {
                "current_close": current.get("close"),
                "previous_close": previous.get("close"),
            }
        )
        mlflow.log_metrics({"predicted_close": prediction})

    return {
        "predicted_next_close": prediction,
        "latency_ms": round(latency_ms, 2),
    }
