"""
visualization.py
-----------------
Plotly chart builders used by the Streamlit dashboard. Every function
returns a go.Figure so app.py can simply st.plotly_chart(fig).

All charts use a shared dark theme (see THEME below) for visual
consistency across the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

THEME = {
    "template": "plotly_dark",
    "bg": "rgba(0,0,0,0)",
    "accent": "#00D4B4",
    "accent2": "#7C4DFF",
    "font_color": "#E8E8E8",
}


def _style(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared dark theme to a figure."""
    fig.update_layout(
        template=THEME["template"],
        title=title,
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["font_color"]),
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def price_vs_demand_chart(sweep_df: pd.DataFrame, current_price: float) -> go.Figure:
    """Line chart of predicted demand across the swept price range."""
    fig = px.line(sweep_df, x="Price", y="Demand", markers=False,
                   color_discrete_sequence=[THEME["accent"]])
    fig.add_vline(x=current_price, line_dash="dash", line_color="white",
                   annotation_text="Current Price")
    return _style(fig, "Price vs Demand")


def price_vs_revenue_chart(sweep_df: pd.DataFrame, current_price: float) -> go.Figure:
    """Line chart of predicted revenue across the swept price range."""
    fig = px.line(sweep_df, x="Price", y="Revenue",
                   color_discrete_sequence=[THEME["accent2"]])
    fig.add_vline(x=current_price, line_dash="dash", line_color="white",
                   annotation_text="Current Price")
    return _style(fig, "Price vs Revenue")


def price_vs_profit_chart(sweep_df: pd.DataFrame, current_price: float) -> go.Figure:
    """Line chart of predicted profit across the swept price range."""
    fig = px.line(sweep_df, x="Price", y="Profit",
                   color_discrete_sequence=["#FFB020"])
    fig.add_vline(x=current_price, line_dash="dash", line_color="white",
                   annotation_text="Current Price")
    return _style(fig, "Price vs Profit")


def sales_forecast_chart(history_df: pd.DataFrame) -> go.Figure:
    """Line chart forecasting next-30-day demand from a simple trend.

    Args:
        history_df: DataFrame with columns 'Day' (1..N) and 'Forecast_Demand'.
    """
    fig = px.area(history_df, x="Day", y="Forecast_Demand",
                   color_discrete_sequence=[THEME["accent"]])
    return _style(fig, "30-Day Sales Forecast")


def feature_importance_chart(importances: pd.Series) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    df = importances.sort_values(ascending=True).reset_index()
    df.columns = ["Feature", "Importance"]
    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Tealgrn")
    return _style(fig, "Feature Importance")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric columns in df."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1)
    return _style(fig, "Correlation Heatmap")


def category_revenue_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total revenue by product category."""
    grouped = df.groupby("Category", as_index=False)["Revenue"].sum()
    grouped = grouped.sort_values("Revenue", ascending=False)
    fig = px.bar(grouped, x="Category", y="Revenue", color="Category",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    return _style(fig, "Category-wise Revenue")


def seasonal_sales_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total demand by season."""
    grouped = df.groupby("Season", as_index=False)["Demand"].sum()
    fig = px.bar(grouped, x="Season", y="Demand", color="Season",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    return _style(fig, "Seasonal Sales")


def competitor_price_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter comparing own price vs competitor price, colored by demand."""
    sample = df.sample(min(len(df), 2000), random_state=42)
    fig = px.scatter(sample, x="Price", y="Competitor_Price", color="Demand",
                      color_continuous_scale="Viridis", opacity=0.7)
    fig.add_shape(type="line", x0=sample["Price"].min(), y0=sample["Price"].min(),
                  x1=sample["Price"].max(), y1=sample["Price"].max(),
                  line=dict(color="white", dash="dot"))
    return _style(fig, "Own Price vs Competitor Price")


def profit_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of profit distribution."""
    fig = px.histogram(df, x="Profit", nbins=50, color_discrete_sequence=["#FFB020"])
    return _style(fig, "Profit Distribution")


def demand_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of demand distribution."""
    fig = px.histogram(df, x="Demand", nbins=50, color_discrete_sequence=[THEME["accent"]])
    return _style(fig, "Demand Distribution")


def price_demand_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of price vs demand colored by category."""
    sample = df.sample(min(len(df), 3000), random_state=42)
    fig = px.scatter(sample, x="Price", y="Demand", color="Category", opacity=0.6)
    return _style(fig, "Price vs Demand (Scatter)")


def profit_box_plot(df: pd.DataFrame) -> go.Figure:
    """Box plot of profit by category."""
    fig = px.box(df, x="Category", y="Profit", color="Category",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    return _style(fig, "Profit Spread by Category")


def price_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of the price column."""
    fig = px.histogram(df, x="Price", nbins=40, color_discrete_sequence=[THEME["accent2"]])
    return _style(fig, "Price Distribution")


def revenue_line_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of average revenue per discount bucket."""
    bucketed = df.copy()
    bucketed["Discount_Bucket"] = (bucketed["Discount"] // 5 * 5).astype(int)
    grouped = bucketed.groupby("Discount_Bucket", as_index=False)["Revenue"].mean()
    fig = px.line(grouped, x="Discount_Bucket", y="Revenue", markers=True,
                  color_discrete_sequence=[THEME["accent"]])
    return _style(fig, "Average Revenue by Discount Level")


def kpi_card_html(label: str, value: str, delta: str | None = None, icon: str = "📊") -> str:
    """Return HTML for a styled KPI card (used inside st.markdown).

    Args:
        label: Card label, e.g. "Predicted Sales".
        value: Main value string to display, already formatted.
        delta: Optional secondary line (e.g. change vs baseline).
        icon: Emoji icon shown in the card.

    Returns:
        HTML string for the card.
    """
    delta_html = f'<div style="font-size:0.8rem;color:#00D4B4;margin-top:4px;">{delta}</div>' if delta else ""
    return f"""
    <div style="
        background: linear-gradient(145deg, #1E1E2E, #262640);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.06);
        text-align:left;
    ">
        <div style="font-size:1.4rem;">{icon}</div>
        <div style="font-size:0.85rem;color:#A0A0B8;margin-top:6px;">{label}</div>
        <div style="font-size:1.4rem;font-weight:700;color:#F0F0F5;margin-top:2px;">{value}</div>
        {delta_html}
    </div>
    """
