from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = (
        model.predict(X_test)
    )

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = {
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    print("\nMetrics")
    print("-" * 40)

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return metrics