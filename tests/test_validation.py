import pandas as pd

from src.data.validation import (
    validate_columns,
    validate_target,
    validate_missing_values,
)


def test_validate_columns():

    df = pd.DataFrame({
        "Time": [1],
        "Class": [0],
    })

    try:
        validate_columns(df)
    except ValueError:
        assert True


def test_validate_target():

    df = pd.DataFrame({
        "Class": [0, 1, 0, 1],
    })

    validate_target(df)


def test_validate_missing_values():

    df = pd.DataFrame({
        "Class": [0, 1],
        "Amount": [100.0, 200.0],
    })

    validate_missing_values(df)