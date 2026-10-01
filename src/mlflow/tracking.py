import mlflow


EXPERIMENT_NAME = "fraud-detection"


def setup_mlflow():

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )