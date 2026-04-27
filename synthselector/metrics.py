"""Evaluation metrics for synthetic data quality assessment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def compute_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Geometric Mean of per-class recalls.

    For binary classification this is ``sqrt(sensitivity * specificity)``.
    For multiclass it is the *n*-th root of the product of all per-class
    recalls, where *n* is the number of classes.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        G-Mean score in [0, 1].
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(np.sqrt(sensitivity * specificity))

    per_class_recall = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall.append(recall)

    return float(np.prod(per_class_recall) ** (1.0 / n_classes))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Return F1 (macro), AUC (macro-OVR), and G-Mean for a single model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    y_prob : array-like of shape (n_samples, n_classes) or None
        Predicted probabilities.  If *None*, AUC is set to 0.

    Returns
    -------
    dict with keys ``"F1"``, ``"AUC"``, ``"GMean"``.
    """
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    auc = 0.0
    if y_prob is not None:
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )

    gmean = compute_gmean(y_true, y_pred)

    return {"F1": f1, "AUC": auc, "GMean": gmean}
