import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import argparse
import os

# ─── Argument Parser (untuk MLProject entry_points) ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth",    type=int, default=10)
args = parser.parse_args()

# ─── MLflow Setup ─────────────────────────────────────────────────────────────
mlflow.set_experiment("Loan Status Experiment")

# ─── Load Data ────────────────────────────────────────────────────────────────
data = pd.read_csv("credit_risk_dataset_preprocessed.csv")

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_example = X_train[:5]

# ─── Training dengan Autolog ──────────────────────────────────────────────────
with mlflow.start_run(run_name="rf_baseline_ci"):
    mlflow.autolog()

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print(f"Test Accuracy: {accuracy:.4f}")