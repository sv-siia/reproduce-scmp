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
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'Confusion Matrix': confusion_matrix(y_true, y_pred)
    }

    if y_scores is not None:
        try:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['ROC AUC'] = 'Undefined (check y_scores or y_true)'

    return metrics
