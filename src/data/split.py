import pandas as pd

from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.20,
    random_state: int = 42,
):

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )