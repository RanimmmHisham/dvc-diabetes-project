# DVC Diabetes Classification Project

This project implements a machine learning pipeline using **DVC (Data Version Control)** to predict diabetes. It utilizes a branch-based experiment workflow to compare a **Logistic Regression** baseline against a **Random Forest** classifier.

## 🚀 Features
* **Data Versioning**: Raw data is tracked via DVC to keep the Git repository lightweight.
* **Pipeline Automation**: A 3-stage pipeline (`preprocess` -> `train` -> `validate`) defined in `dvc.yaml`.
* **Branch-Based Experiments**: Independent model tuning using Git branches (`master` and `random-forest-exp`).
* **Side-by-Side Reports**: Automatic generation of comparison plots and metrics using `dvc plots diff`.

## 🛠️ Project Structure
* **data/**: Contains raw and processed data (tracked by DVC).
* **models/**: Trained model binaries (`model.pkl`).
* **src/**: Python scripts for each pipeline stage (standardized across branches).
* **params.yaml**: Configuration file used to toggle model selection.
* **dvc.yaml**: Defines the pipeline stages, dependencies, and outputs.
* **dvc_plots/**: Generated side-by-side visual reports.

## 🧪 Experiment Workflow
This project follows a standardized experiment structure to ensure results are comparable:
1. **Master Branch**: Contains the Logistic Regression baseline.
2. **random-forest-exp Branch**: Contains the Random Forest experimental setup.

## 📊 Comparison Results

### Visualizing Performance
To generate the side-by-side Confusion Matrix report, run the following command from the `master` branch:

```powershell
dvc plots diff master random-forest-exp
