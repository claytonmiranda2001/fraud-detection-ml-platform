import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the dataset.

    Parameters
    ----------
    df:
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset without exact duplicate rows.
    """

    duplicate_count = df.duplicated().sum()

    print(
        f"Duplicate rows detected: {duplicate_count}"
    )

    df_clean = df.drop_duplicates().copy()

    print(
        f"Rows before cleaning: {len(df)}"
    )

    print(
        f"Rows after cleaning: {len(df_clean)}"
    )

    print(
        f"Rows removed: {len(df) - len(df_clean)}"
    )

    return df_clean
