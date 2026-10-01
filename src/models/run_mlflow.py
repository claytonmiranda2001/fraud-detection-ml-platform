from pathlib import Path

import mlflow
import mlflow.sklearn

from src.data.ingestion import load_dataset
from src.data.validation import validate_dataset
from src.data.split import split_dataset

from src.models.train import create_random_forest_model
from src.models.evaluate import evaluate_model

from src.mlflow.tracking import setup_mlflow
from mlflow.models import infer_signature


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "creditcard.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_random_forest.joblib"
)


def main():

    print("Loading dataset...")

    df = load_dataset(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "Validating dataset..."
    )

    validate_dataset(df)

    print(
        "Splitting dataset..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(df)

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    setup_mlflow()

    with mlflow.start_run(
        run_name="random-forest-tuned"
    ):

        print(
            "Training model..."
        )

        model = create_random_forest_model()

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Evaluating model..."
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        mlflow.log_param(
            "model",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            300,
        )

        mlflow.log_param(
            "max_features",
            "log2",
        )

        mlflow.log_param(
            "min_samples_split",
            5,
        )

        mlflow.log_param(
            "min_samples_leaf",
            1,
        )

        mlflow.log_param(
            "max_depth",
            "None",
        )

        mlflow.log_param(
            "class_weight",
            "balanced",
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"],
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"],
        )

        mlflow.log_metric(
            "f1",
            metrics["f1"],
        )

        mlflow.log_metric(
            "roc_auc",
            metrics["roc_auc"],
        )

        mlflow.log_metric(
            "pr_auc",
            metrics["pr_auc"],
        )

        input_example = X_train.head(1).astype("float64")

        prediction_example = model.predict(input_example)

        signature = infer_signature(
            input_example,
            prediction_example,
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="fraud_random_forest",
            input_example=input_example,
            signature=signature,
            registered_model_name="fraud-random-forest",

        )

        print(
            "\nMLflow run completed."
        )


if __name__ == "__main__":
    main()