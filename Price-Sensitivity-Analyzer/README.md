# 💸 Price Sensitivity Analyzer

A professional, end-to-end data science dashboard that predicts how changing a product's selling price affects **sales, revenue, and profit** — built for a Data Science internship / GitHub portfolio / college presentation.

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![streamlit](https://img.shields.io/badge/streamlit-dashboard-red)

---

## 📖 Project Overview

Retailers and e-commerce sellers constantly face the question: *"What happens to my sales if I change my price?"* This project answers that with a machine learning model trained on product, marketing, and market data, wrapped in an interactive Streamlit dashboard.

Move a price slider and instantly see updated predictions for **demand, revenue, and profit** — plus automatic price optimization, what-if scenario testing, and plain-English business insights.

---

## ✨ Features

- **Interactive price slider** (₹100–₹5,000) with live recalculation of demand, revenue, and profit
- **KPI cards**: current price, predicted sales, revenue, profit, profit margin, demand, average order value
- **Machine learning model comparison**: Linear Regression, Random Forest, Gradient Boosting, and XGBoost (if installed) — automatically selects the best performer by R²
- **15+ interactive Plotly visualizations**: price vs demand/revenue/profit, sales forecast, feature importance, correlation heatmap, category revenue, seasonal sales, competitor comparison, profit/demand distributions, scatter/box/histogram/line charts
- **Price optimization**: revenue-maximizing price, profit-maximizing price, elasticity score, potential revenue/profit gain
- **What-if analysis**: independently adjust price, discount, marketing spend, competitor price, holiday, and season
- **Automated business insights** generated from the model's own price-sensitivity curve
- **Model explainability**: feature importance chart + optional SHAP breakdown of individual predictions
- **Bonus features**: CSV dataset upload, prediction history with CSV export, sample dataset download, 30-day forecast, dark theme, reset button

---

## 🏗️ Architecture

```
User Input (sidebar) → Feature Encoding → Trained Model → Demand Prediction
                                                              │
                          ┌───────────────────────────────────┼──────────────────────────┐
                          ▼                                   ▼                          ▼
                   Revenue/Profit Calc                Price Sweep (100–5000)      SHAP Explainer
                          │                                   │                          │
                          ▼                                   ▼                          ▼
                     KPI Cards                     Optimization + Elasticity      Explainability Tab
                          │                                   │
                          └──────────────────┬────────────────┘
                                             ▼
                                   Plotly Dashboard (Streamlit)
```

**Pipeline:** `dataset/generate_dataset.py` synthesizes realistic transaction data → `train_model.py` (or `notebooks/model_training.ipynb`) trains and compares models, saving the best one to `models/price_model.pkl` → `app.py` loads the model and dataset to power the live dashboard.

---

## 📂 Folder Structure

```
Price-Sensitivity-Analyzer/
│
├── app.py                          # Streamlit dashboard
├── train_model.py                  # Model training & comparison script
├── requirements.txt
├── README.md
├── dataset/
│   ├── generate_dataset.py         # Synthetic dataset generator
│   └── price_sensitivity_dataset.csv
├── notebooks/
│   └── model_training.ipynb        # Exploratory training notebook
├── models/
│   ├── price_model.pkl             # Trained model bundle (model + encoders + metrics)
│   └── metrics.json
├── utils/
│   ├── preprocessing.py            # Cleaning, encoding, feature building
│   ├── prediction.py               # Inference, price sweep, optimization, insights
│   ├── visualization.py            # Plotly chart builders + KPI card HTML
│   └── metrics.py                  # MAE/RMSE/R² evaluation helpers
└── assets/
    └── logo.png
```

---

## ⚙️ Installation

```bash
# 1. Clone or download this project, then move into it
cd Price-Sensitivity-Analyzer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (if not already present)
python dataset/generate_dataset.py

# 5. Train the model
python train_model.py

# 6. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** `xgboost` and `shap` are optional. If either isn't installed, the app and training script gracefully fall back to the remaining models (Linear Regression, Random Forest, Gradient Boosting) and skip the SHAP tab with an informative message.

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError: No module named '_loss'` (or similar unpickling error) when running `streamlit run app.py`:**
The included `models/price_model.pkl` was trained with a specific scikit-learn version. If your installed version differs, retrain locally to regenerate a compatible file:
```bash
python train_model.py
```
This overwrites `models/price_model.pkl` to match your environment, then rerun `streamlit run app.py`.

**PowerShell doesn't support `&&`:** run commands on separate lines instead of chaining with `&&` (PowerShell 5.1 doesn't support it; use `;` or separate lines).

---

## 📸 Screenshots

*(Add screenshots of your running dashboard here — e.g. `assets/screenshot_dashboard.png`, `assets/screenshot_optimization.png`)*

---

## 📊 Results

On the synthetic 15,000-row dataset, model comparison typically yields:

| Model              | MAE   | RMSE  | R²    |
|--------------------|-------|-------|-------|
| Linear Regression  | ~113  | ~148  | ~0.49 |
| Random Forest      | ~31   | ~46   | ~0.95 |
| Gradient Boosting  | ~28   | ~41   | ~0.96 |

Gradient Boosting (or XGBoost, if installed) is selected automatically as the production model based on the highest R² score. Exact numbers are written to `models/metrics.json` after each training run.

---

## 🚀 Future Improvements

- Swap the synthetic dataset for real historical sales data via a connected data warehouse
- Add time-series demand forecasting (e.g. Prophet or ARIMA) alongside the regression model
- Support multi-product portfolio optimization (optimize prices across a whole catalog at once)
- Add authentication and per-user saved scenarios backed by a database
- A/B test recommended price changes against actual outcomes to continuously retrain the model

---

## 👤 Author

Built as a Data Science / Machine Learning portfolio project.

---

## 📄 License

This project is released under the MIT License. Feel free to use, modify, and build on it for personal, academic, or portfolio purposes.
