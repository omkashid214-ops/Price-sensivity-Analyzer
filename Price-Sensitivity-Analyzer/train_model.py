"""
train_model.py
---------------
Trains Linear Regression, Random Forest, Gradient Boosting (and XGBoost,
if installed) models to predict product demand, compares them on
MAE / RMSE / R2, and saves the best-performing model as a bundle at
models/price_model.pkl for use by app.py.

Run:
    python train_model.py
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils.metrics import compare_models, evaluate_model
from utils.preprocessing import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    clean_dataset,
    load_dataset,
)

DATASET_PATH = "dataset/price_sensitivity_dataset.csv"
MODEL_OUT_PATH = "models/price_model.pkl"
METRICS_OUT_PATH = "models/metrics.json"


def get_candidate_models() -> dict:
    """Return the dict of candidate models to train and compare."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            random_state=42, n_jobs=-1
        )
    except ImportError:
        print("xgboost not installed - skipping XGBoost model "
              "(pip install xgboost to enable it).")
    return models


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    raw_df = load_dataset(DATASET_PATH)
    df = clean_dataset(raw_df)
    print(f"Dataset shape after cleaning: {df.shape}")

    X, y, encoders = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_candidate_models()
    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        results[name] = metrics
        fitted_models[name] = model
        print(f"  {name}: {metrics}")

    best_name = compare_models(results)
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} -> {results[best_name]}")

    # Average cost-to-price ratio per category, used by the app to estimate
    # cost price when the user doesn't supply one directly.
    avg_cost_ratio = (
        (df["Cost_Price"] / df["Price"]).groupby(df["Category"]).mean().to_dict()
    )

    bundle = {
        "model": best_model,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": results,
        "best_model_name": best_name,
        "avg_cost_ratio": avg_cost_ratio,
    }

    joblib.dump(bundle, MODEL_OUT_PATH)
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model bundle -> {MODEL_OUT_PATH}")
    print(f"Saved metrics -> {METRICS_OUT_PATH}")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
