**TLDR**: Forget about DVC -- the whole issue is finding a free DVC remote, not paid cloud providers or self-host.
Closest I found is https://dagshub.com/ -- it seems to be for whole git repos (so GitHub-like), with 10 GB storage available which can be used with DVC.
But our instructions say to host code on Github or Gitlab -- so for now going back to Git LFS (Github).

```bash
$ pip install "dvc[s3]"
$ dvc init
# when `dvc add path/to/data.csv` is run, auto stage in git path/to/{.gitignore, data.csv.dvc}
$ dvc config core.autostage true
```

https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data/ -- says "weekly updated data" but last updated 2 months ago

<!-- Gave up on Google Drive remote for DVC:

pip install "dvc[gdrive]"

DVC remote configured: Google Drive folder: https://drive.google.com/drive/u/1/folders/1mSKl9nYIMI_UGbUhgxUuDIL7wJnSrQs3
(StockPredictions_MLOps/ in my IITM GDrive, and shared with our group members - Anish, Gargi, Ambeth)

https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive

```bash
$ dvc remote add gdrive gdrive://1mSKl9nYIMI_UGbUhgxUuDIL7wJnSrQs3         # add remote named "gdrive"; folder id copied from GDrive folder url above
$ dvc remote modify gdrive gdrive_acknowledge_abuse true
$ dvc remote default gdrive
$ dvc push       # opened in browser to authorize -- Google blocked access for DVC app!
```

-->

```bash
pip3 install dagshub --upgrade
dagshub login

# Upload command
dagshub upload sohangchopra/StockPredict_NiftyData "path/to/local/files" --versioning dvc # "target/path/in/repo"

git pull
```

https://dagshub.com/sohangchopra/StockPredict_NiftyData

Gemini:

dvc remote add origin s3://dvc
dvc remote add origin https://dagshub.com/sohangchopra/StockPredict_NiftyData.s3
dvc remote modify origin --local access_key_id <your-dagshub-token>
dvc remote modify origin --local secret_access_key <your-dagshub-token>