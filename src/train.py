import pandas as pd
import joblib
import yaml
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

params = yaml.safe_load(open("params.yaml"))
model_type = params["model"]

train = pd.read_csv("data/processed/train.csv")

X = train.drop("Outcome", axis=1)
y = train["Outcome"]

if model_type == "logistic":
    model = LogisticRegression(max_iter=1000)
elif model_type == "random_forest":
    model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Training complete.")