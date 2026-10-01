from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.

    Parameters
    ----------
    file_path:
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


if __name__ == "__main__":

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "creditcard.csv"
    )

    df = load_dataset(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df["Class"].value_counts())