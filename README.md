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