import json
from pathlib import Path  # Added for path manipulation
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load unique classes from the original features file
    classes_df = pd.read_csv("data/features_iris.csv")
    # Ensure classes are sorted for consistent confusion matrix reporting
    classes = sorted(classes_df["target"].unique().tolist())

    # Load test dataset
    test_dataset = pd.read_csv("data/test.csv")
    y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = test_dataset.drop("target", axis=1).values

    # Load trained model (from local file for this script's logic)
    # If you want to load directly from MLflow, the logic here would change
    clf = joblib.load("models/model.joblib")

    # Make predictions
    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    cm: np.ndarray = confusion_matrix(y, prediction, labels=np.arange(len(classes)))
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")

    return {
        "f1_score": f1,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


if __name__ == "__main__":
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Iris_Experiment")

    # Define path for parent run ID
    parent_run_id_path = Path("data/.mlflow_parent_run_id")
    parent_run_id = None
    if parent_run_id_path.exists():
        with open(parent_run_id_path) as f:
            parent_run_id = f.read().strip()
        print(f"Read parent run ID: {parent_run_id}")

    metrics = evaluate_model()

    # Save metrics locally as JSON
    output_file = "data/eval.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {output_file}")

    # Log metrics and artifacts to MLflow
    with mlflow.start_run(run_name="EvaluationRun") as run:  # Added run_name
        print(f"MLflow Run ID for evaluation: {run.info.run_id}")

        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            print(f"Tagged evaluation run as child of {parent_run_id}")

        mlflow.log_metric("f1_score", metrics["f1_score"])

        # Log confusion matrix as a dictionary (will be saved as json artifact)
        mlflow.log_dict(
            metrics["confusion_matrix"], "evaluation_artifacts/confusion_matrix.json"
        )

        # Log the entire eval.json file as an artifact
        mlflow.log_artifact(output_file, artifact_path="evaluation_artifacts")
        print("Evaluation metrics and artifacts logged to MLflow")
