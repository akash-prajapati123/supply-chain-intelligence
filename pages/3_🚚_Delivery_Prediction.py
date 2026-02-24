"""
🚚 Delivery Prediction Page
Adapted for the DataCo Smart Supply Chain dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data
from src.ml.delivery_prediction import DeliveryPredictor
from src.visualization import charts
import config

st.set_page_config(page_title="Delivery Prediction", page_icon="🚚", layout="wide")

st.markdown("# 🚚 Late Delivery Prediction")
st.markdown("*XGBoost classifier trained on 180K+ DataCo orders for delivery risk assessment*")
st.markdown("---")

df = load_data()

# ─── Train Model ────────────────────────────────────────────────────────────
if st.button("🚀 Train Delivery Prediction Model", use_container_width=True):
    st.session_state["delivery_trained"] = True

if st.session_state.get("delivery_trained", False):
    with st.spinner("🔄 Training XGBoost delivery classifier..."):
        predictor = DeliveryPredictor()
        metrics, y_test, y_pred, y_prob = predictor.train(df)

    # ─── Model Metrics ──────────────────────────────────────────────────
    st.markdown("### 🎯 Model Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")
    m5.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")

    st.markdown("---")

    # ─── Visualizations ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            charts.confusion_matrix_chart(metrics["confusion_matrix"]),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            charts.roc_curve_chart(y_test, y_prob),
            use_container_width=True,
        )

    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    importance = predictor.get_feature_importance(15)
    if not importance.empty:
        st.plotly_chart(
            charts.feature_importance_chart(importance),
            use_container_width=True,
        )

    # ─── What-If Scenario Analysis ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 What-If Scenario Analysis")
    st.markdown("*Predict delivery risk for custom order parameters*")

    with st.form("delivery_prediction_form"):
        wif_col1, wif_col2, wif_col3 = st.columns(3)

        with wif_col1:
            wif_category = st.selectbox("Product Category", config.PRODUCT_CATEGORIES)
            wif_region = st.selectbox("Market Region", config.REGIONS)
            wif_segment = st.selectbox("Customer Segment", config.CUSTOMER_SEGMENTS)

        with wif_col2:
            wif_shipping = st.selectbox("Shipping Mode", config.SHIPPING_MODES)
            wif_department = st.selectbox("Department", config.DEPARTMENTS)

        with wif_col3:
            wif_quantity = st.number_input("Quantity", min_value=1, max_value=5, value=2)
            wif_price = st.number_input("Unit Price ($)", min_value=9.99, max_value=2000.0, value=100.0)

        submitted = st.form_submit_button("🔮 Predict Delivery Risk", use_container_width=True)

        if submitted:
            # Get typical values for this scenario from historical data
            similar = df[
                (df["product_category"] == wif_category) &
                (df["shipping_mode"] == wif_shipping)
            ]
            if len(similar) < 10:
                similar = df

            order_data = {
                "product_category": wif_category,
                "region": wif_region,
                "shipping_mode": wif_shipping,
                "customer_segment": wif_segment,
                "department": wif_department,
                "unit_price": wif_price,
                "quantity": wif_quantity,
                "revenue": wif_price * wif_quantity,
                "discount_percent": similar["discount_percent"].median(),
                "profit_margin": similar["profit_margin"].median(),
                "scheduled_shipping_days": similar["scheduled_shipping_days"].median(),
                "order_month": pd.Timestamp.now().month,
                "order_quarter": pd.Timestamp.now().quarter,
                "order_day_of_week": pd.Timestamp.now().dayofweek,
                "late_delivery": 0,  # Placeholder
            }

            result = predictor.predict_single(order_data)

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            r1, r2, r3 = st.columns(3)
            risk_color = {
                "High": "🔴", "Medium": "🟡", "Low": "🟢"
            }
            r1.metric("Risk Level",
                       f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")
            r2.metric("Late Delivery Probability",
                       f"{result['late_probability']:.1%}")
            r3.metric("On-Time Probability",
                       f"{result['on_time_probability']:.1%}")

else:
    st.info("👆 Click **Train Delivery Prediction Model** to begin!")

    # Show data overview
    st.markdown("### 📊 Current Delivery Overview")
    late_by_mode = df.groupby("shipping_mode")["late_delivery"].mean().reset_index()
    late_by_mode.columns = ["Shipping Mode", "Late Rate"]
    late_by_mode["Late Rate"] = (late_by_mode["Late Rate"] * 100).round(1)

    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(late_by_mode, use_container_width=True)
    with col_b:
        st.metric("Overall Late Delivery Rate", f"{df['late_delivery'].mean():.1%}")
        st.metric("Avg Delay (when late)",
                   f"{df[df['late_delivery']==1]['delivery_delay_days'].mean():.1f} days")
