from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "Time",
    "Amount",
]


def create_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor