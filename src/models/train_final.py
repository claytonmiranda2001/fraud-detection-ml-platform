from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.ingestion import load_dataset
from src.data.validation import validate_dataset
from src.data.split import split_dataset
from src.features.preprocessing import create_preprocessor


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

    print("\nTraining final Random Forest...")

    preprocessor = create_preprocessor()

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features="log2",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()