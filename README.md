# Stock Predictions - ML Ops Project

![Pre-Commit](https://github.com/sohang3112/stock-prediction-mlops/actions/workflows/pre-commit.yaml/badge.svg)

Tech Stack:
- [x] Git + GitHub
- [x] Spark
- [ ] Airflow (pipeline orchestration)
- [x] Mlflow (experiment tracking)
- [x] Git LFS
- [x] FastAPI
- [x] Docker
- [x] Github Actions CI/CD

Other Requirements:
- Logging & Monitoring: at least one of:
  - [ ] ML Drift Detection
  - [ ] ML Performance Monitoring
  - [x] ML API latency / prediction logging
- [ ] better training script: at least 2 models (good enough). Currently only one baseline model exists which is very bad (negative R^2 score!)
- [ ] Ensure training script has Data Preprocessing which:
  * handles missing data (if applicable)
  * cleans data
  * feature engineering (if required)
  * handles class imbalance or data augmentation (if applicable)

Dataset description: Nifty 100 index historical price data of each minute (2015 - 2026)

## Install & Run

One-time setup after cloning repo:

```bash
$ git lfs install
$ pip install pre-commit
$ pre-commit install
```

Main Docker image (for ML prediction) - starts FastAPI server:

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

NOTE: To run training script, I'm using Dev Containers extension in VS Code (which builds & runs Dockerfile and connects VS Code to it as a remote, so directly working in terminal inside it!)

## Helpful Resources

- https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/
