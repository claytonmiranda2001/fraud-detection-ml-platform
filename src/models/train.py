from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import create_preprocessor


def create_logistic_model():

    preprocessor = create_preprocessor()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
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


def create_random_forest_model():

    preprocessor = create_preprocessor()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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

    return pipeline


def train_model(
    X_train,
    y_train,
    model_path: Path,
):

    pipeline = create_logistic_model()

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


def train_random_forest(
    X_train,
    y_train,
    model_path: Path,
):

    pipeline = create_random_forest_model()

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