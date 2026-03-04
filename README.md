# DVC Diabetes Classification Project

This project implements a machine learning pipeline using **DVC (Data Version Control)** to predict diabetes. It compares two different models: **Logistic Regression** and **Random Forest**.

## 🚀 Features
- **Data Versioning:** Raw data is tracked via DVC (`.dvc` files) to keep the Git repository lightweight.
- **Pipeline Automation:** A 3-stage pipeline (`preprocess` -> `train` -> `validate`) defined in `dvc.yaml`.
- **Experiment Tracking:** Multi-model execution supported via `params.yaml`.
- **Visual Reports:** Automatic generation of side-by-side Confusion Matrix plots.

## 🛠️ Project Structure
- `data/`: Contains raw and processed data (tracked by DVC).
- `models/`: Trained model binaries (`.pkl`).
- `src/`: Python scripts for each pipeline stage.
- `params.yaml`: Configuration file for model selection.
- `dvc.yaml`: Defines the pipeline stages and dependencies.

## 🧪 Experiment History
This project followed an iterative development process:
1. **Master Branch:** Initially developed with a Logistic Regression baseline.
2. **`random-forest-exp` Branch:** Created to test a Random Forest Classifier independently.
3. **Multi-Model Integration:** The `random-forest-exp` branch was merged into `master`, and the pipeline was updated to evaluate both models simultaneously in a single run.
   
## 📊 Comparison Results
The pipeline evaluates multiple models in a single run. You can view the performance comparison by running:
```bash
dvc metrics show
dvc plots diff master random-forest-exp
