"""
📦 Inventory Optimization Page
Adapted for the DataCo Smart Supply Chain dataset.
"""
import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data
from src.ml.inventory_optimizer import InventoryOptimizer
from src.visualization import charts

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

st.markdown("# 📦 Inventory Optimization Engine")
st.markdown("*EOQ, Safety Stock, and Reorder Point optimization — computed from DataCo demand patterns*")
st.markdown("---")

df = load_data()

# ─── Run Optimization ──────────────────────────────────────────────────────
optimizer = InventoryOptimizer()

with st.spinner("🔄 Analyzing inventory parameters from demand data..."):
    results = optimizer.analyze_inventory(df)
    recommendations = optimizer.get_recommendations(results)

# ─── Summary KPIs ──────────────────────────────────────────────────────────
st.markdown("### 📊 Optimization Summary")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Categories Analyzed", len(results))
k2.metric("Avg EOQ", f"{results['eoq'].mean():,.0f} units")
k3.metric("Avg Safety Stock (95%)", f"{results['safety_stock_95'].mean():,.0f} units")
k4.metric("Avg Late Delivery Rate", f"{results['late_delivery_rate'].mean():.1f}%")

st.markdown("---")

# ─── Inventory Status Chart ───────────────────────────────────────────────
st.plotly_chart(
    charts.inventory_status(results),
    use_container_width=True,
)

# ─── Detailed Category Table ──────────────────────────────────────────────
st.markdown("### 📋 Category-Level Optimization")

display_cols = [
    "category", "total_orders", "avg_daily_demand", "demand_cv",
    "annual_demand", "eoq",
    "safety_stock_95", "reorder_point_95",
    "safety_stock_99", "reorder_point_99",
    "avg_lead_time", "late_delivery_rate",
]
existing_cols = [c for c in display_cols if c in results.columns]

format_dict = {
    col: fmt for col, fmt in {
        "avg_daily_demand": "{:.1f}",
        "demand_cv": "{:.1f}",
        "annual_demand": "{:,.0f}",
        "eoq": "{:,.0f}",
        "total_orders": "{:,}",
        "safety_stock_95": "{:,.0f}",
        "reorder_point_95": "{:,.0f}",
        "safety_stock_99": "{:,.0f}",
        "reorder_point_99": "{:,.0f}",
        "avg_lead_time": "{:.1f}",
        "late_delivery_rate": "{:.1f}",
    }.items() if col in existing_cols
}

styled = results[existing_cols].style.format(format_dict)
if "late_delivery_rate" in existing_cols:
    styled = styled.background_gradient(subset=["late_delivery_rate"], cmap="YlOrRd")
if "demand_cv" in existing_cols:
    styled = styled.background_gradient(subset=["demand_cv"], cmap="YlOrRd")

st.dataframe(styled, use_container_width=True, height=400)

# ─── Recommendations ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔔 Recommendations")

if recommendations:
    for rec in recommendations:
        severity_icon = {
            "High": "🔴", "Medium": "🟡", "Low": "🟢"
        }.get(rec.get("priority", "Medium"), "⚪")

        with st.expander(
            f"{rec['category']} — {rec['action'][:60]}... ({severity_icon} {rec.get('priority', 'Medium')})"
        ):
            st.markdown(f"**Action:** {rec['action']}")
            st.markdown(f"**Reason:** {rec['reason']}")
else:
    st.success("✅ No critical inventory issues detected!")

# ─── EOQ Calculator ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧮 Interactive EOQ Calculator")

with st.form("eoq_calculator"):
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        calc_demand = st.number_input("Annual Demand (units)", min_value=100, value=10000)
    with ec2:
        calc_order_cost = st.number_input("Ordering Cost ($)", min_value=1.0, value=50.0)
    with ec3:
        calc_hold_rate = st.slider("Holding Cost Rate (%)", min_value=5, max_value=50, value=20)
    with ec4:
        calc_unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=100.0)

    calc_submitted = st.form_submit_button("📊 Calculate EOQ", use_container_width=True)

    if calc_submitted:
        eoq_result = optimizer.calculate_eoq(
            calc_demand, calc_order_cost, calc_unit_price, calc_hold_rate / 100
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("💡 Optimal Order Quantity", f"{eoq_result:,.0f} units")
        r2.metric("📅 Orders/Year", f"{calc_demand / max(eoq_result, 1):,.1f}")
        r3.metric("📅 Reorder Every",
                   f"{365 / max(calc_demand / max(eoq_result, 1), 1):,.0f} days")

# ─── Safety Stock Calculator ──────────────────────────────────────────────
st.markdown("### 🛡️ Safety Stock Calculator")

with st.form("safety_stock_calculator"):
    ss1, ss2, ss3 = st.columns(3)
    with ss1:
        ss_demand = st.number_input("Avg Daily Demand", min_value=1.0, value=50.0)
        ss_demand_std = st.number_input("Demand Std Dev", min_value=0.1, value=15.0)
    with ss2:
        ss_lead = st.number_input("Avg Lead Time (days)", min_value=1.0, value=4.0)
        ss_lead_std = st.number_input("Lead Time Std Dev", min_value=0.1, value=1.0)
    with ss3:
        ss_service = st.slider("Service Level (%)", min_value=80, max_value=99, value=95)

    ss_submitted = st.form_submit_button("🛡️ Calculate Safety Stock", use_container_width=True)

    if ss_submitted:
        ss_result = optimizer.calculate_safety_stock(
            ss_demand, ss_demand_std, ss_lead, ss_lead_std, ss_service / 100
        )

        sr1, sr2, sr3 = st.columns(3)
        sr1.metric("🛡️ Safety Stock", f"{ss_result['safety_stock']:,.0f} units")
        sr2.metric("📍 Reorder Point", f"{ss_result['reorder_point']:,.0f} units")
        sr3.metric("📐 Z-Score", f"{ss_result['z_score']:.2f}")
