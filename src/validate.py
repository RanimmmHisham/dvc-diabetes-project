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
models = params["models"]

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

metrics = {}

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
if len(models) == 1:
    axes = [axes]

for i, model_name in enumerate(models):
    model = joblib.load(f"models/model_{model_name}.pkl")
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    metrics[model_name] = {"accuracy": acc}
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(model_name)

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()

print("Done! Metrics saved and confusion matrices plotted.")