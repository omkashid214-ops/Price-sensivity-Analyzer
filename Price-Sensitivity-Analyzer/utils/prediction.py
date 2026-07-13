"""
prediction.py
--------------
Inference-time helpers: loading the trained model bundle, running
single/batch predictions, price-sweep optimization, and elasticity
calculations used by the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import FEATURE_COLUMNS, encode_categoricals


@dataclass
class ModelBundle:
    """Container for everything needed to run inference."""
    model: object
    encoders: dict
    feature_columns: list
    metrics: Dict[str, Dict[str, float]]
    best_model_name: str
    avg_cost_ratio: Dict[str, float]  # category -> avg cost/price ratio


def load_model_bundle(path: str = "models/price_model.pkl") -> ModelBundle:
    """Load the trained model bundle from disk.

    Args:
        path: Path to the joblib-serialized bundle.

    Returns:
        A ModelBundle instance.
    """
    data = joblib.load(path)
    return ModelBundle(**data)


def make_input_row(
    price: float,
    marketing_spend: float,
    competitor_price: float,
    discount: float,
    season: str,
    holiday: int,
    category: str,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching FEATURE_COLUMNS.

    Args:
        price: Selling price.
        marketing_spend: Marketing budget.
        competitor_price: Competitor's price for a similar product.
        discount: Discount percentage (0-100).
        season: One of the known season categories.
        holiday: 1 if holiday period, else 0.
        category: Product category.

    Returns:
        Single-row DataFrame with columns in FEATURE_COLUMNS order.
    """
    row = pd.DataFrame([{
        "Price": price,
        "Marketing_Spend": marketing_spend,
        "Competitor_Price": competitor_price,
        "Discount": discount,
        "Season": season,
        "Holiday": holiday,
        "Category": category,
    }])
    return row[FEATURE_COLUMNS]


def predict_demand(bundle: ModelBundle, input_df: pd.DataFrame) -> np.ndarray:
    """Predict demand for one or more input rows.

    Args:
        bundle: Loaded ModelBundle.
        input_df: DataFrame with FEATURE_COLUMNS.

    Returns:
        Array of predicted demand values (non-negative).
    """
    encoded_df, _ = encode_categoricals(input_df.copy(), bundle.encoders)
    preds = bundle.model.predict(encoded_df[bundle.feature_columns])
    return np.clip(preds, 0, None)


def price_sweep(
    bundle: ModelBundle,
    base_inputs: dict,
    cost_price: float,
    price_min: float = 100,
    price_max: float = 5000,
    steps: int = 100,
) -> pd.DataFrame:
    """Sweep price across a range and compute demand, revenue, and profit.

    Args:
        bundle: Loaded ModelBundle.
        base_inputs: Dict with keys marketing_spend, competitor_price,
            discount, season, holiday, category (everything except price).
        cost_price: Unit cost price used to compute profit.
        price_min: Minimum price in the sweep.
        price_max: Maximum price in the sweep.
        steps: Number of points in the sweep.

    Returns:
        DataFrame with columns Price, Demand, Revenue, Profit.
    """
    prices = np.linspace(price_min, price_max, steps)
    rows = pd.DataFrame([{**base_inputs, "Price": p} for p in prices])
    rows = rows[FEATURE_COLUMNS]
    demand = predict_demand(bundle, rows)
    revenue = prices * demand
    profit = revenue - (cost_price * demand)
    return pd.DataFrame({
        "Price": prices,
        "Demand": demand,
        "Revenue": revenue,
        "Profit": profit,
    })


def find_optimal_prices(sweep_df: pd.DataFrame) -> Dict[str, float]:
    """Find revenue-maximizing and profit-maximizing prices from a sweep.

    Args:
        sweep_df: Output of price_sweep().

    Returns:
        Dict with keys: max_revenue_price, max_revenue, max_profit_price,
        max_profit.
    """
    rev_idx = sweep_df["Revenue"].idxmax()
    profit_idx = sweep_df["Profit"].idxmax()
    return {
        "max_revenue_price": round(float(sweep_df.loc[rev_idx, "Price"]), 2),
        "max_revenue": round(float(sweep_df.loc[rev_idx, "Revenue"]), 2),
        "max_profit_price": round(float(sweep_df.loc[profit_idx, "Price"]), 2),
        "max_profit": round(float(sweep_df.loc[profit_idx, "Profit"]), 2),
    }


def price_elasticity(sweep_df: pd.DataFrame, current_price: float) -> float:
    """Estimate point price elasticity of demand near the current price.

    Elasticity = (%change in demand) / (%change in price), evaluated using
    the two sweep points that straddle current_price.

    Args:
        sweep_df: Output of price_sweep(), sorted by Price ascending.
        current_price: The price at which to evaluate elasticity.

    Returns:
        Elasticity value (typically negative; magnitude > 1 = elastic).
    """
    df = sweep_df.sort_values("Price").reset_index(drop=True)
    idx = (df["Price"] - current_price).abs().idxmin()
    idx = min(max(idx, 1), len(df) - 2)

    p1, p2 = df.loc[idx - 1, "Price"], df.loc[idx + 1, "Price"]
    d1, d2 = df.loc[idx - 1, "Demand"], df.loc[idx + 1, "Demand"]

    if p1 == p2 or d1 == 0:
        return 0.0

    pct_change_demand = (d2 - d1) / d1
    pct_change_price = (p2 - p1) / p1
    if pct_change_price == 0:
        return 0.0
    return round(pct_change_demand / pct_change_price, 3)


def generate_insights(
    sweep_df: pd.DataFrame,
    current_price: float,
    current_demand: float,
    elasticity: float,
    optimal: Dict[str, float],
) -> list[str]:
    """Generate plain-English business insight strings.

    Args:
        sweep_df: Output of price_sweep().
        current_price: Currently selected price.
        current_demand: Predicted demand at the current price.
        elasticity: Output of price_elasticity().
        optimal: Output of find_optimal_prices().

    Returns:
        List of human-readable insight strings.
    """
    insights = []

    demand_change_pct = elasticity * 10  # effect of a 10% price change
    direction = "decreases" if demand_change_pct < 0 else "increases"
    insights.append(
        f"Increasing price by 10% {direction} demand by approximately "
        f"{abs(demand_change_pct):.1f}%."
    )

    insights.append(
        f"Revenue peaks around ₹{optimal['max_revenue_price']:.0f}, yielding "
        f"an estimated revenue of ₹{optimal['max_revenue']:,.0f}."
    )

    insights.append(
        f"Highest profit occurs at approximately ₹{optimal['max_profit_price']:.0f}, "
        f"yielding an estimated profit of ₹{optimal['max_profit']:,.0f}."
    )

    if optimal["max_profit_price"] > current_price:
        insights.append(
            "Your current price is below the profit-optimal point — there may be "
            "room to increase price without significantly hurting demand."
        )
    elif optimal["max_profit_price"] < current_price:
        insights.append(
            "Your current price is above the profit-optimal point — a small "
            "reduction could increase overall profit."
        )

    elasticity_label = "elastic (price-sensitive)" if abs(elasticity) > 1 else "inelastic (less price-sensitive)"
    insights.append(f"Demand at this price point is {elasticity_label} (elasticity ≈ {elasticity}).")

    return insights
