from pathlib import Path  # Added for path manipulation

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Iris_Experiment")

    # Define path for parent run ID
    parent_run_id_path = Path("data/.mlflow_parent_run_id")

    # Load train set
    train_dataset = pd.read_csv("data/train.csv")

    # Get X and Y
    y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = train_dataset.drop("target", axis=1).values

    # Model parameters
    params = {"C": 0.01, "solver": "lbfgs", "max_iter": 100}

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(**params)

    with mlflow.start_run(run_name="TrainingRun") as run:  # Added run_name
        parent_run_id = run.info.run_id
        print(f"MLflow Run ID (Training): {parent_run_id}")

        # Save the parent run ID for the evaluation script
        parent_run_id_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure data dir exists
        with open(parent_run_id_path, "w") as f:
            f.write(parent_run_id)
        print(f"Saved parent run ID to {parent_run_id_path}")

        mlflow.log_params(params)

        clf.fit(X, y)

        # Log the scikit-learn model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="iris_model",
            registered_model_name="iris-logistic-regression",
        )
        print("Model logged to MLflow")

        # Save model locally as well (optional, if you still need it)
        joblib.dump(clf, "models/model.joblib")
        print("Model saved locally to models/model.joblib")
