from pathlib import Path

from src.data.cleaning import (
    remove_duplicates,
)

from src.data.ingestion import (
    load_dataset,
)

from src.data.validation import (
    validate_dataset,
)

from src.data.split import (
    split_dataset,
)

from src.models.train import (
    train_model,
)

from src.models.evaluate import (
    evaluate_model,
)


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
    / "fraud_model.joblib"
)


def main():

    print("Loading dataset...")

    df = load_dataset(
        DATA_PATH
    )

    print(
        f"Dataset shape before cleaning: {df.shape}"
    )

    print(
        "Removing duplicate rows..."
    )

    df = remove_duplicates(df)

    print(
        f"Dataset shape after cleaning: {df.shape}"
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

    print(
        "Training model..."
    )

    model = train_model(
        X_train,
        y_train,
        MODEL_PATH,
    )

    print(
        "Evaluating model..."
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()