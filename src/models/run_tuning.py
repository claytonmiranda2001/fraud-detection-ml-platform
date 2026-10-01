from pathlib import Path

from src.data.ingestion import load_dataset
from src.data.validation import validate_dataset
from src.data.split import split_dataset

from src.models.tune import tune_random_forest


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

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)

    best_model = tune_random_forest(
        X_train,
        y_train,
    )

    print("\nTuning completed.")

    print("\nBest model:")
    print(best_model)


if __name__ == "__main__":
    main()