"""Fastapi server - main entry point. ONLY FOR deployment (model prediction), NOT training."""

from fastapi import FastAPI

# import airflow
# import pyspark
# import mlflow

app = FastAPI()


@app.get("/")
def hello():
    return "Hello World"
