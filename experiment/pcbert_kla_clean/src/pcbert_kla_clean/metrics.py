from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Rec": recall_score(y_true, y_pred, zero_division=0),
        "Pre": precision_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "SP": specificity,
        "AUPRC": average_precision_score(y_true, y_score),
    }


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "MCC",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 901,
) -> tuple[float, dict[str, float]]:
    """Find a decision threshold using validation labels and probabilities."""
    if metric not in {"ACC", "Rec", "Pre", "MCC", "F1", "SP"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_score, threshold=best_threshold)
    best_score = best_metrics[metric]

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_score, threshold=float(threshold))
        score = metrics[metric]
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
            best_metrics = metrics

    return best_threshold, best_metrics
