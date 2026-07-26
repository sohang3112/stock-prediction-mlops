"""Fastapi server - main entry point."""
import airflow
import mlflow
import pyspark
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def hello():
    return 'Hello World'