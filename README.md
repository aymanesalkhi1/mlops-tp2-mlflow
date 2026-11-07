# Fraud pipeline and MLflow registry

Fraud detection split into three DVC stages (prepare, train, evaluate), then an MLflow registry: the model is logged, tagged with the `staging` and `production` aliases, and can be rolled back.

MLflow runs locally at `http://127.0.0.1:5000`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 5000
```

In a second terminal, with the environment activated:

```bash
python donnees.py
dvc repro
python src/enregistrer.py
python src/alias.py
```

`src/rollback.py` points an alias back to an earlier version.
