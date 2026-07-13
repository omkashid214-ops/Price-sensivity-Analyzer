"""
metrics.py
----------
Model evaluation helpers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R^2 for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dict with keys 'MAE', 'RMSE', 'R2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}


def compare_models(results: Dict[str, Dict[str, float]]) -> str:
    """Pick the best model name from a results dict based on highest R2.

    Args:
        results: Mapping of model_name -> metrics dict (must contain 'R2').

    Returns:
        Name of the best-performing model.
    """
    return max(results, key=lambda name: results[name]["R2"])
