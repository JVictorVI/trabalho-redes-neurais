"""Métricas de teste e interpretação dos erros."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[truth, pred] += 1
    return matrix


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    """Calcula matriz de confusão e métricas precision, recall e F1 por classe."""

    cm = confusion_matrix(y_true, y_pred, len(classes))
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        predicted = cm[:, i].sum()
        actual = cm[i, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(actual),
            }
        )
    report = pd.DataFrame(rows)
    accuracy = float(np.trace(cm) / cm.sum())
    macro_f1 = float(report["f1"].mean())
    weighted_f1 = float(np.average(report["f1"], weights=report["support"]))
    summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    return report, cm, summary


def top_confusions(cm: np.ndarray, classes: list[str], limit: int = 8) -> list[dict[str, Any]]:
    """Lista os pares de classes mais confundidos, ignorando os acertos da diagonal."""

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    items: list[dict[str, Any]] = []
    for truth, pred in zip(*np.unravel_index(np.argsort(off_diag.ravel())[::-1], cm.shape)):
        count = int(off_diag[truth, pred])
        if count == 0 or len(items) >= limit:
            break
        items.append(
            {
                "true_class": classes[truth],
                "predicted_class": classes[pred],
                "count": count,
            }
        )
    return items
