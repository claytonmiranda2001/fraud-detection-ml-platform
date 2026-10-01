from pathlib import Path

from src.data.ingestion import load_dataset
from src.data.validation import validate_dataset
from src.data.split import split_dataset

from src.models.train import (
    train_model,
    train_random_forest,
)

from src.models.evaluate import evaluate_model


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

LOGISTIC_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest.joblib"
)


def main():

    print("Loading dataset...")

    df = load_dataset(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    print("Validating dataset...")

    validate_dataset(df)

    print("Splitting dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(df)

    # --------------------------------
    # Logistic Regression
    # --------------------------------

    print("\n")
    print("=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    logistic_model = train_model(
        X_train,
        y_train,
        LOGISTIC_MODEL_PATH,
    )

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test,
        y_test,
    )

    # --------------------------------
    # Random Forest
    # --------------------------------

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    random_forest_model = train_random_forest(
        X_train,
        y_train,
        RANDOM_FOREST_MODEL_PATH,
    )

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
    )

    # --------------------------------
    # Comparison
    # --------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<15}"
        f"{'Logistic':<15}"
        f"{'Random Forest':<15}"
    )

    print("-" * 45)

    for metric in logistic_metrics:

        print(
            f"{metric:<15}"
            f"{logistic_metrics[metric]:<15.4f}"
            f"{random_forest_metrics[metric]:<15.4f}"
        )


if __name__ == "__main__":
    main()