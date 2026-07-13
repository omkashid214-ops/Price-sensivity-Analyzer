"""
generate_dataset.py
--------------------
Generates a realistic synthetic e-commerce pricing dataset for the
Price Sensitivity Analyzer project.

Encodes the following real-world relationships:
    - Higher price          -> lower demand      (price elasticity)
    - Higher marketing spend -> higher demand
    - Higher discount        -> higher demand
    - Holiday periods         -> higher demand
    - Profit = Revenue - (Cost Price * Units Sold)

Run:
    python generate_dataset.py
Produces:
    price_sensitivity_dataset.csv  (15,000 rows) in this folder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 15_000

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Groceries"]
SEASONS = ["Winter", "Summer", "Monsoon", "Spring"]

# Base price range and category-specific elasticity / base demand multipliers.
CATEGORY_PROFILE = {
    "Electronics":    {"base_price": (2000, 5000), "cost_ratio": 0.65, "elasticity": 1.4, "base_demand": 220},
    "Fashion":        {"base_price": (400, 2500),   "cost_ratio": 0.45, "elasticity": 1.8, "base_demand": 300},
    "Home & Kitchen": {"base_price": (500, 3500),   "cost_ratio": 0.55, "elasticity": 1.2, "base_demand": 250},
    "Beauty":         {"base_price": (150, 1500),   "cost_ratio": 0.35, "elasticity": 2.0, "base_demand": 350},
    "Sports":         {"base_price": (300, 3000),   "cost_ratio": 0.50, "elasticity": 1.3, "base_demand": 200},
    "Groceries":      {"base_price": (100, 800),    "cost_ratio": 0.70, "elasticity": 0.8, "base_demand": 500},
}


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the synthetic price-sensitivity dataset.

    Args:
        n_rows: Number of rows (product-transactions) to generate.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with the columns described in the project spec.
    """
    rng = np.random.default_rng(seed)

    categories = rng.choice(CATEGORIES, size=n_rows)
    seasons = rng.choice(SEASONS, size=n_rows)
    holiday = rng.choice([0, 1], size=n_rows, p=[0.75, 0.25])
    product_id = np.array([f"P{100000 + i}" for i in range(n_rows)])

    price = np.zeros(n_rows)
    cost_price = np.zeros(n_rows)
    demand = np.zeros(n_rows)
    discount = rng.uniform(0, 40, size=n_rows).round(1)          # percent
    marketing_spend = rng.uniform(500, 50000, size=n_rows).round(2)
    competitor_price = np.zeros(n_rows)

    for cat, profile in CATEGORY_PROFILE.items():
        mask = categories == cat
        n = mask.sum()
        low, high = profile["base_price"]

        cat_price = rng.uniform(low, high, size=n)
        price[mask] = cat_price
        cost_price[mask] = cat_price * profile["cost_ratio"] * rng.uniform(0.9, 1.1, size=n)
        competitor_price[mask] = cat_price * rng.uniform(0.85, 1.15, size=n)

        # Normalize price within category range to get a 0-1 "price level"
        price_level = (cat_price - low) / (high - low + 1e-9)

        elasticity = profile["elasticity"]
        base_demand = profile["base_demand"]

        season_boost = np.select(
            [seasons[mask] == "Winter", seasons[mask] == "Summer",
             seasons[mask] == "Monsoon", seasons[mask] == "Spring"],
            [1.05, 1.10, 0.95, 1.0],
            default=1.0,
        )

        holiday_boost = np.where(holiday[mask] == 1, 1.35, 1.0)
        discount_boost = 1 + (discount[mask] / 100) * 1.2
        marketing_boost = 1 + (marketing_spend[mask] / marketing_spend.max()) * 0.6
        competitor_effect = 1 + np.clip(
            (competitor_price[mask] - cat_price) / cat_price, -0.3, 0.3
        ) * 0.8  # cheaper than competitor -> more demand

        # Core elasticity effect: demand falls as price_level rises
        elasticity_effect = np.exp(-elasticity * price_level)

        noise = rng.normal(1.0, 0.08, size=n)

        raw_demand = (
            base_demand
            * elasticity_effect
            * season_boost
            * holiday_boost
            * discount_boost
            * marketing_boost
            * competitor_effect
            * noise
        )
        demand[mask] = np.clip(raw_demand, 5, None).round().astype(int)

    price = price.round(2)
    cost_price = cost_price.round(2)
    competitor_price = competitor_price.round(2)
    demand = demand.astype(int)

    revenue = (price * demand).round(2)
    profit = (revenue - cost_price * demand).round(2)

    df = pd.DataFrame({
        "Product_ID": product_id,
        "Category": categories,
        "Price": price,
        "Cost_Price": cost_price,
        "Competitor_Price": competitor_price,
        "Discount": discount,
        "Marketing_Spend": marketing_spend,
        "Season": seasons,
        "Holiday": holiday,
        "Demand": demand,
        "Revenue": revenue,
        "Profit": profit,
    })

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    out_path = "price_sensitivity_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Generated {len(dataset):,} rows -> {out_path}")
    print(dataset.describe(include="all").T)
