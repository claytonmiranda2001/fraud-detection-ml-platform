import pandas as pd

from src.data.cleaning import remove_duplicates


def test_remove_duplicates():

    df = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": [10, 10, 20],
        }
    )

    result = remove_duplicates(df)

    assert len(result) == 2


def test_remove_duplicates_preserves_unique_rows():

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [10, 20, 30],
        }
    )

    result = remove_duplicates(df)

    assert len(result) == 3