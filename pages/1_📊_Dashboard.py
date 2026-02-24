"""
📊 Dashboard – Supply Chain KPIs and Overview
Adapted for the DataCo Smart Supply Chain dataset (180K+ orders).
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import (
    load_data, get_kpi_metrics, get_monthly_trends,
    get_category_performance, get_region_performance,
    get_shipping_performance, filter_data
)
from src.visualization import charts

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ─── Load Data ──────────────────────────────────────────────────────────────
df = load_data()

# ─── Sidebar Filters ───────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["order_date"].min().date(), df["order_date"].max().date()),
    min_value=df["order_date"].min().date(),
    max_value=df["order_date"].max().date(),
)

selected_categories = st.sidebar.multiselect(
    "Product Categories",
    options=sorted(df["product_category"].unique()),
    default=[],
)

selected_regions = st.sidebar.multiselect(
    "Market Regions",
    options=sorted(df["region"].unique()),
    default=[],
)

selected_shipping = st.sidebar.multiselect(
    "Shipping Modes",
    options=sorted(df["shipping_mode"].unique()),
    default=[],
)

selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=sorted(df["customer_segment"].unique()),
    default=[],
)

# Apply filters
filtered_df = filter_data(
    df,
    date_range=date_range if len(date_range) == 2 else None,
    categories=selected_categories or None,
    regions=selected_regions or None,
    shipping_modes=selected_shipping or None,
    segments=selected_segments or None,
)

# ─── Page Header ────────────────────────────────────────────────────────────
st.markdown("# 📊 Supply Chain Dashboard")
st.markdown(f"*Showing **{len(filtered_df):,}** orders out of **{len(df):,}** total  •  "
            f"DataCo Supply Chain Dataset (2015–2018)*")
st.markdown("---")

# ─── KPI Metrics ────────────────────────────────────────────────────────────
kpi = get_kpi_metrics(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"${kpi['total_revenue']:,.0f}")
col2.metric("📦 Total Orders", f"{kpi['total_orders']:,}")
col3.metric("📈 Total Profit", f"${kpi['total_profit']:,.0f}")
col4.metric("💵 Avg Order Value", f"${kpi['avg_order_value']:,.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("✅ On-Time Rate", f"{kpi['on_time_rate']:.1f}%")
col6.metric("🚚 Avg Shipping Days", f"{kpi['avg_shipping_days']:.1f}")
col7.metric("⚠️ Fraud Rate", f"{kpi['fraud_rate']:.2f}%")
col8.metric("❌ Cancellation Rate", f"{kpi['cancellation_rate']:.2f}%")

st.markdown("---")

# ─── Revenue & Order Trends ────────────────────────────────────────────────
monthly = get_monthly_trends(filtered_df)

tab1, tab2 = st.tabs(["📈 Revenue Trend", "📦 Order Volume"])

with tab1:
    st.plotly_chart(charts.revenue_trend(monthly), use_container_width=True)

with tab2:
    st.plotly_chart(charts.orders_trend(monthly), use_container_width=True)

# ─── Category & Regional Analysis ──────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    cat_perf = get_category_performance(filtered_df)
    tab_a, tab_b = st.tabs(["Revenue by Category", "Category Distribution"])
    with tab_a:
        st.plotly_chart(charts.category_revenue(cat_perf), use_container_width=True)
    with tab_b:
        st.plotly_chart(charts.category_pie(cat_perf), use_container_width=True)

with col_right:
    regional = get_region_performance(filtered_df)
    st.plotly_chart(charts.regional_bar(regional), use_container_width=True)

    st.plotly_chart(charts.delivery_performance(filtered_df), use_container_width=True)

# ─── Shipping Analysis ─────────────────────────────────────────────────────
st.markdown("### 🚚 Shipping Mode Analysis")
st.plotly_chart(charts.shipping_mode_analysis(filtered_df), use_container_width=True)

# ─── Detailed Data ─────────────────────────────────────────────────────────
with st.expander("📋 Raw Data Sample", expanded=False):
    display_cols = [
        "order_date", "product_category", "product_name", "region",
        "customer_segment", "shipping_mode", "quantity", "unit_price",
        "revenue", "profit", "late_delivery", "actual_shipping_days",
        "delivery_status", "order_status",
    ]
    existing_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[existing_cols].head(100),
        use_container_width=True,
        height=400,
    )
