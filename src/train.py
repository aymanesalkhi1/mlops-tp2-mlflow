import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Detection_Fraude")

params = yaml.safe_load(open("params.yaml"))["train"]

train = pd.read_csv("data/train.csv")
X_train = train.drop(columns=["fraude"])
y_train = train["fraude"]

with mlflow.start_run(run_name="RandomForest_Base"):
    mlflow.log_params(params)
    
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    clf.fit(X_train, y_train)
    
    joblib.dump(clf, "models/model.pkl")
    mlflow.sklearn.log_model(
        clf, 
        artifact_path="model", 
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print("Modèle entraîné et enregistré dans MLflow")
