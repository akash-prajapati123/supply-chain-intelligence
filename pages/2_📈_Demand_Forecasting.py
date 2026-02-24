"""
📈 Demand Forecasting Page
"""
import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data
from src.ml.demand_forecasting import DemandForecaster
from src.visualization import charts

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")

st.markdown("# 📈 ML-Powered Demand Forecasting")
st.markdown("*XGBoost regression with time-series features, lag variables, and seasonality encoding*")
st.markdown("---")

# Load data
df = load_data()

# ─── Controls ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    selected_category = st.selectbox(
        "Product Category",
        options=["All Categories"] + sorted(df["product_category"].unique().tolist()),
        index=0,
    )

with col2:
    forecast_horizon = st.slider(
        "Forecast Horizon (days)", min_value=7, max_value=180, value=90, step=7
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    train_button = st.button("🚀 Train Model & Forecast", use_container_width=True)

# ─── Model Training ────────────────────────────────────────────────────────
if train_button or "demand_trained" in st.session_state:
    category = None if selected_category == "All Categories" else selected_category

    with st.spinner("🔄 Training XGBoost demand forecasting model..."):
        forecaster = DemandForecaster()
        metrics, y_test, y_pred, test_idx = forecaster.train(df, category)
        st.session_state["demand_trained"] = True

    # ─── Model Metrics ──────────────────────────────────────────────────
    st.markdown("### 🎯 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['mae']:.1f}")
    m2.metric("RMSE", f"{metrics['rmse']:.1f}")
    m3.metric("R² Score", f"{metrics['r2']:.3f}")
    m4.metric("MAPE", f"{metrics['mape']:.1f}%")

    st.markdown("---")

    # ─── Forecast ───────────────────────────────────────────────────────
    with st.spinner("🔮 Generating forecast..."):
        forecast_df = forecaster.forecast_future(df, forecast_horizon, category)

    # Prepare historical data for chart
    cat_data = df if category is None else df[df["product_category"] == category]
    historical = cat_data.groupby(
        cat_data["order_date"].dt.date
    )["quantity"].sum().reset_index()
    historical.columns = ["order_date", "demand"]
    historical["order_date"] = pd.to_datetime(historical["order_date"])

    # Forecast Chart
    st.markdown("### 🔮 Demand Forecast")
    st.plotly_chart(
        charts.forecast_chart(historical, forecast_df),
        use_container_width=True,
    )

    # ─── Forecast Summary ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Forecast Statistics")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Category** | {selected_category} |
        | **Forecast Period** | {forecast_horizon} days |
        | **Avg Daily Demand** | {forecast_df['predicted_demand'].mean():,.0f} units |
        | **Peak Demand** | {forecast_df['predicted_demand'].max():,.0f} units |
        | **Min Demand** | {forecast_df['predicted_demand'].min():,.0f} units |
        | **Total Predicted** | {forecast_df['predicted_demand'].sum():,.0f} units |
        | **Peak Date** | {forecast_df.loc[forecast_df['predicted_demand'].idxmax(), 'date'].strftime('%Y-%m-%d')} |
        """)

    with col_b:
        st.markdown("### 🎯 Feature Importance")
        importance = forecaster.get_feature_importance(15)
        if not importance.empty:
            st.plotly_chart(
                charts.feature_importance_chart(importance),
                use_container_width=True,
            )

    # ─── Forecast Data ─────────────────────────────────────────────────
    with st.expander("📋 Forecast Data Table"):
        st.dataframe(
            forecast_df.style.format({"predicted_demand": "{:.0f}"}),
            use_container_width=True,
            height=300,
        )

else:
    st.info("👆 Select a category and click **Train Model & Forecast** to begin!")

    # Show preview
    st.markdown("### 📊 Historical Demand Overview")
    daily_all = df.groupby(df["order_date"].dt.date)["quantity"].sum().reset_index()
    daily_all.columns = ["order_date", "demand"]
    daily_all["order_date"] = pd.to_datetime(daily_all["order_date"])

    import plotly.express as px
    fig = px.line(
        daily_all, x="order_date", y="demand",
        title="Daily Demand (All Categories)",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
