from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.features.preprocessing import create_preprocessor


def tune_random_forest(X_train, y_train):

    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [
            100,
            200,
            300,
            500,
        ],
        "model__max_depth": [
            None,
            10,
            20,
            30,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
        ],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    print("\nBest Random Forest parameters:")
    print(search.best_params_)

    print(
        f"\nBest CV PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    return search.best_estimator_