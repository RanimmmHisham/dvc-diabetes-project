import pandas as pd
import joblib
import json
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

model = joblib.load("models/model.pkl")

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

with open("metrics/metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=4)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(colorbar=False)
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
print("Done! Plot saved to plots/confusion_matrix.png")