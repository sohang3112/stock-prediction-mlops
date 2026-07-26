# Stock Predictions - ML Ops Project

![Pre-Commit](https://github.com/sohang3112/stock-prediction-mlops/actions/workflows/pre-commit.yaml/badge.svg)

Tech Stack & Other Project Requirements:
- [x] Git + GitHub
- [ ] Spark (actual ML training)
- [ ] Airflow (pipeline orchestration)
- [ ] Mlflow (experiment tracking)
- [x] Git LFS
- [x] FastAPI
- [x] Docker
- [x] Github Actions CI/CD

Dataset description: Nifty 100 index historical price data of each minute (2015 - 2026)

## Install

```bash
$ pip install pre-commit
$ pre-commit install
$ pre-commit run --all-files
```

```bash
$ docker build -t stock_prediction .
$ docker container run --publish 8000:8000 stock_prediction

 ⚡️ Starting FastAPI in production mode
 
 🐍 Using import string: app:app (auto-discovered, use --verbose to learn more)
 
 💡 You can configure an entrypoint in pyproject.toml for this app with:
 
    [tool.fastapi]
    entrypoint = "app:app"
 
 🌐 Server started at http://0.0.0.0:8000
    Documentation at http://0.0.0.0:8000/docs
```

```bash
git lfs install
git lfs track **/*.csv    # track data (NIFTY 100_minute.csv is 55 MB)
```

## Helpful Resources

- https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/