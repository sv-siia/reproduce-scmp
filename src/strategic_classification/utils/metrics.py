from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(y_true, y_pred, y_scores=None):
    """
    Computes evaluation metrics for binary classification.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_scores: Optional, prediction scores or probabilities for ROC AUC

    Returns:
    - Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 'Undefined (check y_scores or y_true)'

    return metrics


if __name__ == "__main__":
    # test usage
    y_true = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    y_scores = [0.1, 0.9, 0.8, 0.4, 0.3, 0.2, 0.85, 0.6, 0.95, 0.05]

    response = evaluate_model(y_true, y_pred, y_scores)
    print(response)
