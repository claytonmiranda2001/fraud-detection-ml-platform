from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    create_preprocessor,
)


def create_model():

    preprocessor = (
        create_preprocessor()
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def train_model(
    X_train,
    y_train,
    model_path: Path,
):

    pipeline = create_model()

    pipeline.fit(
        X_train,
        y_train,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    return pipeline