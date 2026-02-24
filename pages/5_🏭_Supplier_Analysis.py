"""
🏭 Department / Supplier Analysis Page
Adapted for the DataCo Smart Supply Chain dataset.
Since DataCo has no explicit supplier IDs, departments are scored as supply units.
"""
import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data
from src.ml.supplier_scoring import SupplierScorer
from src.visualization import charts

st.set_page_config(page_title="Department Analysis", page_icon="🏭", layout="wide")

st.markdown("# 🏭 Department Intelligence & Risk Scoring")
st.markdown("*Multi-criteria weighted scoring across delivery, profitability, reliability, cost efficiency, and diversity*")
st.markdown("---")

df = load_data()

# ─── Score Departments ──────────────────────────────────────────────────────
scorer = SupplierScorer()

with st.spinner("🔄 Analyzing department performance..."):
    scores = scorer.score_suppliers(df)
    risk_summary = scorer.get_risk_summary(scores)
    suggestions = scorer.get_improvement_suggestions(scores)

# ─── Summary KPIs ──────────────────────────────────────────────────────────
st.markdown("### 📊 Department Portfolio Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Departments", risk_summary.get("total_entities", 0))
k2.metric("Avg Score", f"{risk_summary.get('avg_score', 0):.3f}")
k3.metric("🏆 Best Department", risk_summary.get("best", "N/A"))
k4.metric("⚠️ Worst Department", risk_summary.get("worst", "N/A"))

# Risk distribution
risk_dist = risk_summary.get("risk_distribution", {})
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("🟢 Very Low Risk", risk_dist.get("Very Low", 0))
r2.metric("🔵 Low Risk", risk_dist.get("Low", 0))
r3.metric("🟡 Medium Risk", risk_dist.get("Medium", 0))
r4.metric("🟠 High Risk", risk_dist.get("High", 0))
r5.metric("🔴 Critical Risk", risk_dist.get("Critical", 0))

st.markdown("---")

# ─── Department Comparison Chart ───────────────────────────────────────────
st.plotly_chart(
    charts.supplier_comparison(scores),
    use_container_width=True,
)

# ─── Department Detail Cards ───────────────────────────────────────────────
st.markdown("### 📡 Individual Department Analysis")

selected_supplier = st.selectbox(
    "Select Department for Detailed Analysis",
    options=scores["supplier"].tolist(),
    index=0,
)

supplier_row = scores[scores["supplier"] == selected_supplier].iloc[0]

col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(
        charts.supplier_radar(supplier_row),
        use_container_width=True,
    )

with col2:
    grade_color = {
        "A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"
    }
    risk_color = {
        "Very Low": "🟢", "Low": "🔵", "Medium": "🟡", "High": "🟠", "Critical": "🔴"
    }

    grade = str(supplier_row["grade"])
    risk = str(supplier_row["risk_level"])

    st.markdown(f"""
    ### {supplier_row['supplier']}

    | Metric | Value |
    |--------|-------|
    | **Overall Score** | {supplier_row['overall_score']:.3f} |
    | **Grade** | {grade_color.get(grade, '')} {grade} |
    | **Risk Level** | {risk_color.get(risk, '')} {risk} |
    | **Total Orders** | {supplier_row['total_orders']:,} |
    | **Total Revenue** | ${supplier_row['total_revenue']:,.2f} |
    | **Avg Shipping Days** | {supplier_row['avg_shipping_days']:.1f} days |
    | **Late Delivery Rate** | {supplier_row['late_delivery_rate']:.1%} |

    #### Performance Breakdown
    | Dimension | Score |
    |-----------|-------|
    | Delivery Performance | {supplier_row['delivery_performance']:.2f} |
    | Profitability | {supplier_row['profitability']:.2f} |
    | Volume Capability | {supplier_row['volume_capability']:.2f} |
    | Order Reliability | {supplier_row['order_reliability']:.2f} |
    | Cost Efficiency | {supplier_row['cost_efficiency']:.2f} |
    | Product Diversity | {supplier_row['product_diversity']:.2f} |
    """)

# ─── Scoring Weights ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚖️ Scoring Methodology")

weight_df = pd.DataFrame([
    {"Dimension": k.replace("_", " ").title(), "Weight": f"{v:.0%}"}
    for k, v in scorer.WEIGHTS.items()
])
st.dataframe(weight_df, use_container_width=True, height=250)

# ─── Full Leaderboard ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏆 Department Leaderboard")

display_cols = [
    "supplier", "overall_score", "grade", "risk_level",
    "delivery_performance", "profitability", "volume_capability",
    "order_reliability", "cost_efficiency", "product_diversity",
    "total_orders", "total_revenue",
]
existing_cols = [c for c in display_cols if c in scores.columns]

st.dataframe(
    scores[existing_cols].style.format({
        "overall_score": "{:.3f}",
        "delivery_performance": "{:.2f}",
        "profitability": "{:.2f}",
        "volume_capability": "{:.2f}",
        "order_reliability": "{:.2f}",
        "cost_efficiency": "{:.2f}",
        "product_diversity": "{:.2f}",
        "total_orders": "{:,.0f}",
        "total_revenue": "${:,.0f}",
    }).background_gradient(subset=["overall_score"], cmap="RdYlGn"),
    use_container_width=True,
    height=500,
)

# ─── Improvement Suggestions ──────────────────────────────────────────────
if suggestions:
    st.markdown("---")
    st.markdown("### 💡 Improvement Suggestions")

    for sug in suggestions[:15]:
        risk_icon = {
            "Very Low": "🟢", "Low": "🔵", "Medium": "🟡",
            "High": "🟠", "Critical": "🔴"
        }.get(str(sug.get("risk_level", "Medium")), "⚪")

        with st.expander(
            f"{risk_icon} {sug['supplier']} (Grade {sug['grade']}) — Weak: {sug['weakest_area']}"
        ):
            st.markdown(f"**Suggestion:** {sug['suggestion']}")
