import pandas as pd
import joblib
import json
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

params = yaml.safe_load(open("params.yaml"))
model_name = params["models"][0]
model = joblib.load("models/model.pkl")

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

metrics = {model_name: {"accuracy": acc}}
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(colorbar=False)
plt.title(f"Confusion Matrix: {model_name}")

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

print(f"Done! {model_name} metrics saved and plot generated.")