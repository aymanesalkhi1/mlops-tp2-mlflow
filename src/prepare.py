import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/transactions.csv")
train, test = train_test_split(
    df, 
    test_size=params["test_size"], 
    random_state=params["random_state"], 
    stratify=df["fraude"]
)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("Données préparées avec succès !")
