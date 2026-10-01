import pandas as pd


EXPECTED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]


def validate_columns(df: pd.DataFrame) -> None:

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


def validate_target(df: pd.DataFrame) -> None:

    if "Class" not in df.columns:
        raise ValueError(
            "Target column 'Class' not found."
        )

    unique_values = set(
        df["Class"].unique()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )


def validate_missing_values(
    df: pd.DataFrame,
) -> None:

    missing_values = df.isnull().sum()

    columns_with_missing = (
        missing_values[
            missing_values > 0
        ]
    )

    if not columns_with_missing.empty:

        raise ValueError(
            "Missing values detected:\n"
            f"{columns_with_missing}"
        )


def validate_dataset(
    df: pd.DataFrame,
) -> None:

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    validate_columns(df)
    validate_target(df)
    validate_missing_values(df)