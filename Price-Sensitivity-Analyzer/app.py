"""
app.py
------
Price Sensitivity Analyzer - Streamlit dashboard.

A professional business-analytics dashboard that predicts how changing a
product's selling price affects demand, revenue, and profit, with
interactive what-if analysis, price optimization, and model explainability.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.prediction import (
    find_optimal_prices,
    generate_insights,
    load_model_bundle,
    make_input_row,
    predict_demand,
    price_elasticity,
    price_sweep,
)
from utils.preprocessing import clean_dataset
from utils.visualization import (
    category_revenue_chart,
    competitor_price_chart,
    correlation_heatmap,
    demand_distribution_chart,
    feature_importance_chart,
    kpi_card_html,
    price_demand_scatter,
    price_histogram,
    price_vs_demand_chart,
    price_vs_profit_chart,
    price_vs_revenue_chart,
    profit_box_plot,
    profit_distribution_chart,
    revenue_line_chart,
    sales_forecast_chart,
    seasonal_sales_chart,
)

DATASET_PATH = "dataset/price_sensitivity_dataset.csv"
MODEL_PATH = "models/price_model.pkl"

st.set_page_config(
    page_title="Price Sensitivity Analyzer",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global dark theme styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #14141f, #0B0B12 60%);
        color: #E8E8E8;
    }
    section[data-testid="stSidebar"] {
        background-color: #14141f;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1E1E2E, #262640);
        border-radius: 14px;
        padding: 10px 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    h1, h2, h3 { color: #F0F0F5; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data / model loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_dataset(df)


@st.cache_resource(show_spinner=False)
def get_model_bundle(path: str):
    return load_model_bundle(path)


def ensure_artifacts_exist() -> bool:
    """Check dataset/model exist; show setup instructions if not."""
    missing = []
    if not os.path.exists(DATASET_PATH):
        missing.append(f"- Dataset not found at `{DATASET_PATH}`. Run `python dataset/generate_dataset.py`.")
    if not os.path.exists(MODEL_PATH):
        missing.append(f"- Model not found at `{MODEL_PATH}`. Run `python train_model.py`.")
    if missing:
        st.error("Setup required before the dashboard can run:\n\n" + "\n".join(missing))
        return False
    return True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("💸 Price Sensitivity Analyzer")
    st.sidebar.caption("Configure your scenario, then explore the dashboard.")

    uploaded = st.sidebar.file_uploader(
        "Upload your own dataset (optional)", type=["csv"],
        help="Must contain the same columns as price_sensitivity_dataset.csv",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario Inputs")

    category = st.sidebar.selectbox("Category", sorted(df["Category"].unique()))
    season = st.sidebar.selectbox("Season", sorted(df["Season"].unique()))
    holiday = st.sidebar.toggle("Holiday period", value=False)

    price_min, price_max = 100, 5000
    price = st.sidebar.slider(
        "Price (₹)", min_value=float(price_min), max_value=float(price_max),
        value=float(np.clip(df[df["Category"] == category]["Price"].median(), price_min, price_max)),
        step=10.0,
    )
    discount = st.sidebar.slider("Discount (%)", 0.0, 40.0, 10.0, step=1.0)
    marketing_spend = st.sidebar.slider(
        "Marketing Spend (₹)", 500.0, 50000.0,
        float(df["Marketing_Spend"].median()), step=500.0,
    )
    competitor_price = st.sidebar.slider(
        "Competitor Price (₹)", 100.0, 6000.0,
        float(df[df["Category"] == category]["Competitor_Price"].median()), step=10.0,
    )

    st.sidebar.markdown("---")
    reset = st.sidebar.button("🔄 Reset to Defaults")

    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "⬇️ Download Sample Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="price_sensitivity_dataset.csv",
        mime="text/csv",
    )

    return {
        "uploaded": uploaded,
        "category": category,
        "season": season,
        "holiday": 1 if holiday else 0,
        "price": price,
        "discount": discount,
        "marketing_spend": marketing_spend,
        "competitor_price": competitor_price,
        "reset": reset,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("💸 Price Sensitivity Analyzer")
    st.caption(
        "Predict how changing your selling price affects sales, revenue, "
        "and profit — powered by machine learning."
    )

    if not ensure_artifacts_exist():
        st.stop()

    df = get_dataset(DATASET_PATH)
    bundle = get_model_bundle(MODEL_PATH)

    inputs = render_sidebar(df)

    if inputs["uploaded"] is not None:
        try:
            df = clean_dataset(pd.read_csv(inputs["uploaded"]))
            st.success("Custom dataset loaded and applied to the charts below.")
        except Exception as e:
            st.warning(f"Could not read uploaded file, using default dataset instead ({e}).")

    # Cost price estimate for the selected category (from training data ratios)
    cost_ratio = bundle.avg_cost_ratio.get(inputs["category"], 0.5)
    cost_price = inputs["price"] * cost_ratio

    # --- Core prediction for current scenario ---
    input_row = make_input_row(
        price=inputs["price"],
        marketing_spend=inputs["marketing_spend"],
        competitor_price=inputs["competitor_price"],
        discount=inputs["discount"],
        season=inputs["season"],
        holiday=inputs["holiday"],
        category=inputs["category"],
    )
    predicted_demand = float(predict_demand(bundle, input_row)[0])
    revenue = inputs["price"] * predicted_demand
    profit = revenue - cost_price * predicted_demand
    margin = (profit / revenue * 100) if revenue > 0 else 0
    avg_order_value = inputs["price"] * (1 - inputs["discount"] / 100)

    # --- KPI Cards ---
    st.subheader("📊 Key Metrics")
    kpi_cols = st.columns(7)
    kpi_values = [
        ("💰", "Current Price", f"₹{inputs['price']:,.0f}"),
        ("📦", "Predicted Sales", f"{predicted_demand:,.0f} units"),
        ("💵", "Revenue", f"₹{revenue:,.0f}"),
        ("📈", "Profit", f"₹{profit:,.0f}"),
        ("🎯", "Profit Margin", f"{margin:,.1f}%"),
        ("📊", "Demand", f"{predicted_demand:,.0f}"),
        ("🛒", "Avg Order Value", f"₹{avg_order_value:,.0f}"),
    ]
    for col, (icon, label, value) in zip(kpi_cols, kpi_values):
        with col:
            st.markdown(kpi_card_html(label, value, icon=icon), unsafe_allow_html=True)

    st.markdown("---")

    # --- Price sweep (used by several charts + optimization) ---
    base_inputs = {
        "Marketing_Spend": inputs["marketing_spend"],
        "Competitor_Price": inputs["competitor_price"],
        "Discount": inputs["discount"],
        "Season": inputs["season"],
        "Holiday": inputs["holiday"],
        "Category": inputs["category"],
    }
    with st.spinner("Recalculating price sensitivity curve..."):
        sweep_df = price_sweep(
            bundle, base_inputs, cost_price=cost_price,
            price_min=100, price_max=5000, steps=120,
        )

    # --- Price Optimization ---
    st.subheader("🧭 Price Optimization")
    optimal = find_optimal_prices(sweep_df)
    elasticity = price_elasticity(sweep_df, inputs["price"])
    revenue_gain = optimal["max_revenue"] - revenue
    profit_gain = optimal["max_profit"] - profit

    opt_cols = st.columns(4)
    opt_cols[0].markdown(kpi_card_html(
        "Max Revenue Price", f"₹{optimal['max_revenue_price']:,.0f}", icon="🚀"), unsafe_allow_html=True)
    opt_cols[1].markdown(kpi_card_html(
        "Max Profit Price", f"₹{optimal['max_profit_price']:,.0f}", icon="🏆"), unsafe_allow_html=True)
    opt_cols[2].markdown(kpi_card_html(
        "Elasticity Score", f"{elasticity:.2f}", icon="⚖️"), unsafe_allow_html=True)
    opt_cols[3].markdown(kpi_card_html(
        "Potential Profit Gain", f"₹{profit_gain:,.0f}", icon="✨"), unsafe_allow_html=True)

    st.markdown("---")

    # --- Business Insights ---
    st.subheader("💡 Business Insights")
    insights = generate_insights(sweep_df, inputs["price"], predicted_demand, elasticity, optimal)
    for insight in insights:
        st.markdown(f"- {insight}")

    st.markdown("---")

    # --- Visualizations ---
    st.subheader("📈 Visual Analysis")

    tab1, tab2, tab3 = st.tabs(["Price Impact", "Dataset Insights", "Model Explainability"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(price_vs_demand_chart(sweep_df, inputs["price"]), use_container_width=True)
        c2.plotly_chart(price_vs_revenue_chart(sweep_df, inputs["price"]), use_container_width=True)

        c3, c4 = st.columns(2)
        c3.plotly_chart(price_vs_profit_chart(sweep_df, inputs["price"]), use_container_width=True)

        forecast_df = pd.DataFrame({
            "Day": np.arange(1, 31),
            "Forecast_Demand": np.clip(
                predicted_demand + np.cumsum(np.random.default_rng(1).normal(0, 3, 30)),
                0, None,
            ),
        })
        c4.plotly_chart(sales_forecast_chart(forecast_df), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.plotly_chart(category_revenue_chart(df), use_container_width=True)
        c2.plotly_chart(seasonal_sales_chart(df), use_container_width=True)

        c3, c4 = st.columns(2)
        c3.plotly_chart(competitor_price_chart(df), use_container_width=True)
        c4.plotly_chart(profit_distribution_chart(df), use_container_width=True)

        c5, c6 = st.columns(2)
        c5.plotly_chart(demand_distribution_chart(df), use_container_width=True)
        c6.plotly_chart(price_demand_scatter(df), use_container_width=True)

        c7, c8 = st.columns(2)
        c7.plotly_chart(profit_box_plot(df), use_container_width=True)
        c8.plotly_chart(price_histogram(df), use_container_width=True)

        c9, c10 = st.columns(2)
        c9.plotly_chart(revenue_line_chart(df), use_container_width=True)
        c10.plotly_chart(correlation_heatmap(df), use_container_width=True)

    with tab3:
        st.markdown(f"**Best model:** {bundle.best_model_name}")
        metrics_df = pd.DataFrame(bundle.metrics).T
        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

        if hasattr(bundle.model, "feature_importances_"):
            importances = pd.Series(
                bundle.model.feature_importances_, index=bundle.feature_columns
            )
            st.plotly_chart(feature_importance_chart(importances), use_container_width=True)
        else:
            st.info("The selected best model does not expose feature importances directly.")

        st.markdown("#### SHAP Explainability")
        try:
            import shap

            encoded_row, _ = (input_row, None)
            explainer = shap.Explainer(bundle.model)
            shap_values = explainer(input_row[bundle.feature_columns])
            st.write("SHAP values for the current scenario:")
            shap_df = pd.DataFrame({
                "Feature": bundle.feature_columns,
                "SHAP Value": shap_values.values[0],
            }).sort_values("SHAP Value", key=abs, ascending=False)
            st.dataframe(shap_df, use_container_width=True)
        except ImportError:
            st.info("Install `shap` (`pip install shap`) to see individual prediction "
                     "explanations and SHAP summary plots here.")
        except Exception as e:
            st.info(f"SHAP explanation unavailable for this model/input combination: {e}")

    st.markdown("---")

    # --- What-if Analysis ---
    st.subheader("🔮 What-if Analysis")
    st.caption("Adjust any factor below to see instantly updated predictions, "
               "independent of the sidebar scenario.")

    wc1, wc2, wc3 = st.columns(3)
    wi_price = wc1.slider("What-if Price (₹)", 100.0, 5000.0, inputs["price"], step=10.0, key="wi_price")
    wi_discount = wc2.slider("What-if Discount (%)", 0.0, 40.0, inputs["discount"], step=1.0, key="wi_discount")
    wi_marketing = wc3.slider("What-if Marketing Spend (₹)", 500.0, 50000.0,
                               inputs["marketing_spend"], step=500.0, key="wi_marketing")

    wc4, wc5, wc6 = st.columns(3)
    wi_competitor = wc4.slider("What-if Competitor Price (₹)", 100.0, 6000.0,
                                inputs["competitor_price"], step=10.0, key="wi_competitor")
    wi_holiday = wc5.toggle("What-if Holiday", value=bool(inputs["holiday"]), key="wi_holiday")
    wi_season = wc6.selectbox("What-if Season", sorted(df["Season"].unique()),
                               index=sorted(df["Season"].unique()).index(inputs["season"]), key="wi_season")

    wi_row = make_input_row(
        price=wi_price, marketing_spend=wi_marketing, competitor_price=wi_competitor,
        discount=wi_discount, season=wi_season, holiday=1 if wi_holiday else 0,
        category=inputs["category"],
    )
    wi_demand = float(predict_demand(bundle, wi_row)[0])
    wi_revenue = wi_price * wi_demand
    wi_profit = wi_revenue - cost_price * wi_demand

    wi_cols = st.columns(3)
    wi_cols[0].metric("Predicted Demand", f"{wi_demand:,.0f} units",
                       delta=f"{wi_demand - predicted_demand:+.0f}")
    wi_cols[1].metric("Predicted Revenue", f"₹{wi_revenue:,.0f}",
                       delta=f"₹{wi_revenue - revenue:+,.0f}")
    wi_cols[2].metric("Predicted Profit", f"₹{wi_profit:,.0f}",
                       delta=f"₹{wi_profit - profit:+,.0f}")

    st.markdown("---")

    # --- Prediction history / export ---
    st.subheader("🗂️ Prediction History & Export")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("➕ Save Current Scenario to History"):
        st.session_state.history.append({
            "Category": inputs["category"], "Season": inputs["season"],
            "Holiday": inputs["holiday"], "Price": inputs["price"],
            "Discount": inputs["discount"], "Marketing_Spend": inputs["marketing_spend"],
            "Competitor_Price": inputs["competitor_price"],
            "Predicted_Demand": round(predicted_demand, 1),
            "Revenue": round(revenue, 2), "Profit": round(profit, 2),
        })

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Export Prediction History (CSV)", data=csv_buffer.getvalue(),
            file_name="prediction_history.csv", mime="text/csv",
        )
    else:
        st.caption("No saved scenarios yet — click the button above to start building history.")

    st.markdown("---")
    st.caption(
        "Built with Streamlit, scikit-learn, and Plotly · "
        "Price Sensitivity Analyzer — Data Science Portfolio Project"
    )


if __name__ == "__main__":
    main()
