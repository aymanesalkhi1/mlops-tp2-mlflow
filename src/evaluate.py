import yaml
import pandas as pd
import mlflow
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Detection_Fraude")

test = pd.read_csv("data/test.csv")
X_test = test.drop(columns=["fraude"])
y_test = test["fraude"]

clf = joblib.load("models/model.pkl")
preds = clf.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "precision": float(precision_score(y_test, preds)),
    "recall": float(recall_score(y_test, preds)),
    "f1_score": float(f1_score(y_test, preds))
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("reports/confusion_matrix.png")

with mlflow.start_run(run_name="Evaluation"):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("reports/confusion_matrix.png")

print("Évaluation terminée.")
