import pandas as pd
import joblib
import json
import os
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
    json.dump({"accuracy": acc}, f)

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("plots/confusion_matrix.png")

print("Validation complete.")