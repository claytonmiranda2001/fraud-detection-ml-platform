from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.ingestion import load_dataset
from src.data.validation import validate_dataset
from src.data.split import split_dataset
from src.features.preprocessing import create_preprocessor
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
    print("TRAINING TUNED RANDOM FOREST")
    print("=" * 60)

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
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features="log2",
                    max_depth=None,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train,
    )

    print("\n")
    print("=" * 60)
    print("TUNED RANDOM FOREST - TEST SET")
    print("=" * 60)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nFinal test metrics:")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )


if __name__ == "__main__":
    main()