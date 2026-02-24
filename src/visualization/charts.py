"""
Plotly Chart Library for Supply Chain Intelligence Platform.
Consistent, premium-quality visualizations.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",     # Indigo
    "secondary": "#8B5CF6",   # Violet
    "success": "#10B981",     # Emerald
    "warning": "#F59E0B",     # Amber
    "danger": "#EF4444",      # Red
    "info": "#06B6D4",        # Cyan
    "accent": "#EC4899",      # Pink
    "neutral": "#6B7280",     # Gray
}

PALETTE = [
    "#6366F1", "#8B5CF6", "#EC4899", "#06B6D4",
    "#10B981", "#F59E0B", "#EF4444", "#F97316",
    "#14B8A6", "#A855F7",
]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E5E7EB"),
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(
        bgcolor="rgba(17,24,39,0.9)",
        font_size=13,
        font_family="Inter, sans-serif",
    ),
)


def _apply_layout(fig, title="", height=400):
    """Apply consistent layout to figures."""
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color="#F9FAFB")),
        height=height,
    )
    return fig


def revenue_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend with area fill."""
    fig = go.Figure()
    rev_col = "revenue" if "revenue" in monthly_df.columns else "total_revenue"
    fig.add_trace(go.Scatter(
        x=monthly_df["order_date"],
        y=monthly_df[rev_col],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color=COLORS["primary"], width=2.5),
        name="Revenue",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    return _apply_layout(fig, "📈 Monthly Revenue Trend")


def orders_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Monthly order volume trend."""
    fig = go.Figure()
    ord_col = "orders" if "orders" in monthly_df.columns else "total_orders"
    fig.add_trace(go.Bar(
        x=monthly_df["order_date"],
        y=monthly_df[ord_col],
        marker=dict(
            color=monthly_df[ord_col],
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        ),
        hovertemplate="<b>%{x|%b %Y}</b><br>Orders: %{y:,}<extra></extra>",
    ))
    return _apply_layout(fig, "📦 Monthly Order Volume")


def category_revenue(cat_df: pd.DataFrame) -> go.Figure:
    """Revenue by category horizontal bar chart."""
    fig = go.Figure()
    rev_col = "revenue" if "revenue" in cat_df.columns else "total_revenue"
    top_cats = cat_df.head(20)  # Show top 20 to avoid overcrowding
    fig.add_trace(go.Bar(
        y=top_cats["product_category"],
        x=top_cats[rev_col],
        orientation="h",
        marker=dict(color=PALETTE * 3),
        hovertemplate=(
            "<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>"
        ),
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, "🏷️ Revenue by Category (Top 20)", height=550)


def category_pie(cat_df: pd.DataFrame) -> go.Figure:
    """Category distribution donut chart."""
    rev_col = "revenue" if "revenue" in cat_df.columns else "total_revenue"
    top_cats = cat_df.head(10)  # Top 10 for pie chart readability
    fig = go.Figure(go.Pie(
        labels=top_cats["product_category"],
        values=top_cats[rev_col],
        hole=0.55,
        marker=dict(colors=PALETTE[:len(top_cats)]),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    return _apply_layout(fig, "📊 Top 10 Category Distribution", height=450)


def regional_map(regional_df: pd.DataFrame) -> go.Figure:
    """Regional performance bar chart."""
    fig = go.Figure()
    rev_col = "revenue" if "revenue" in regional_df.columns else "total_revenue"
    fig.add_trace(go.Bar(
        x=regional_df["region"],
        y=regional_df[rev_col],
        marker=dict(
            color=regional_df[rev_col],
            colorscale="Viridis",
        ),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    return _apply_layout(fig, "🌍 Revenue by Market Region")


def regional_bar(regional_df: pd.DataFrame) -> go.Figure:
    """Alias for regional_map for Dashboard compatibility."""
    return regional_map(regional_df)


def delivery_performance(df: pd.DataFrame) -> go.Figure:
    """On-time vs late delivery gauge chart."""
    on_time = (1 - df["late_delivery"].mean()) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=on_time,
        delta=dict(reference=85, valueformat=".1f"),
        number=dict(suffix="%", font=dict(size=36)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=COLORS["primary"]),
            steps=[
                dict(range=[0, 60], color="rgba(239,68,68,0.2)"),
                dict(range=[60, 80], color="rgba(245,158,11,0.2)"),
                dict(range=[80, 100], color="rgba(16,185,129,0.2)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["success"], width=3),
                thickness=0.8, value=85,
            ),
        ),
    ))
    return _apply_layout(fig, "⏱️ On-Time Delivery Rate", height=350)


def shipping_mode_analysis(df: pd.DataFrame) -> go.Figure:
    """Shipping mode distribution and late delivery rate."""
    mode_data = df.groupby("shipping_mode").agg(
        count=("order_id", "count"),
        late_rate=("late_delivery", "mean"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=mode_data["shipping_mode"],
        y=mode_data["count"],
        name="Orders",
        marker_color=COLORS["primary"],
        opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=mode_data["shipping_mode"],
        y=mode_data["late_rate"] * 100,
        name="Late Rate %",
        mode="lines+markers",
        line=dict(color=COLORS["danger"], width=3),
        marker=dict(size=10),
    ), secondary_y=True)

    fig.update_yaxes(title_text="Order Count", secondary_y=False)
    fig.update_yaxes(title_text="Late Delivery %", secondary_y=True)
    return _apply_layout(fig, "🚚 Shipping Mode Analysis")


def forecast_chart(
    historical_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> go.Figure:
    """Demand forecast with historical data and predictions."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_df["order_date"],
        y=historical_df["demand"],
        mode="lines",
        name="Historical Demand",
        line=dict(color=COLORS["primary"], width=1.5),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_demand"],
        mode="lines",
        name="Forecast",
        line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(236,72,153,0.1)",
    ))

    # Confidence band (±15%)
    upper = forecast_df["predicted_demand"] * 1.15
    lower = forecast_df["predicted_demand"] * 0.85

    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(236,72,153,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band (±15%)",
        showlegend=True,
    ))

    return _apply_layout(fig, "🔮 Demand Forecast", height=500)


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Feature importance horizontal bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df["feature"],
        x=importance_df["importance"],
        orientation="h",
        marker=dict(
            color=importance_df["importance"],
            colorscale=[[0, COLORS["info"]], [1, COLORS["primary"]]],
        ),
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, "🎯 Feature Importance", height=450)


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Confusion matrix heatmap."""
    labels = ["On-Time", "Late"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18),
        colorscale=[[0, "rgba(99,102,241,0.1)"], [1, COLORS["primary"]]],
        showscale=False,
    ))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return _apply_layout(fig, "🔄 Confusion Matrix", height=400)


def supplier_radar(supplier_row: pd.Series) -> go.Figure:
    """Spider/Radar chart for individual department/supplier."""
    categories = [
        "Delivery", "Profitability", "Volume",
        "Reliability", "Cost Eff.", "Diversity"
    ]
    values = [
        supplier_row["delivery_performance"],
        supplier_row.get("profitability", 0),
        supplier_row.get("volume_capability", 0),
        supplier_row.get("order_reliability", 0),
        supplier_row.get("cost_efficiency", 0),
        supplier_row.get("product_diversity", 0),
    ]
    values.append(values[0])  # Close the polygon

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color=COLORS["primary"], width=2),
        name=supplier_row["supplier"],
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
    )
    return _apply_layout(fig, f"📡 {supplier_row['supplier']}", height=400)


def supplier_comparison(scores_df: pd.DataFrame) -> go.Figure:
    """Supplier overall scores comparison."""
    fig = go.Figure()
    colors = [
        COLORS["success"] if s >= 0.7
        else COLORS["warning"] if s >= 0.5
        else COLORS["danger"]
        for s in scores_df["overall_score"]
    ]
    fig.add_trace(go.Bar(
        x=scores_df["supplier"],
        y=scores_df["overall_score"],
        marker=dict(color=colors),
        hovertemplate=(
            "<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["success"],
                  annotation_text="Good (0.7)")
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Acceptable (0.5)")
    return _apply_layout(fig, "🏆 Supplier Score Comparison", height=450)


def inventory_status(inv_df: pd.DataFrame) -> go.Figure:
    """Inventory optimization status by category."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "EOQ vs Safety Stock (95%)", "Late Delivery Rate by Category"
    ))

    top = inv_df.head(15)  # Show top 15 categories

    fig.add_trace(go.Bar(
        x=top["category"],
        y=top["eoq"],
        name="EOQ",
        marker_color=COLORS["primary"],
        opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=top["category"],
        y=top["safety_stock_95"],
        name="Safety Stock (95%)",
        marker_color=COLORS["info"],
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=top["category"],
        y=top["late_delivery_rate"],
        name="Late %",
        marker_color=[
            COLORS["danger"] if r > 55
            else COLORS["warning"] if r > 45
            else COLORS["success"]
            for r in top["late_delivery_rate"]
        ],
    ), row=1, col=2)

    return _apply_layout(fig, "📦 Inventory Optimization Status", height=450)


def roc_curve_chart(y_test, y_prob) -> go.Figure:
    """ROC Curve visualization."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.3f})",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color=COLORS["neutral"], width=1, dash="dash"),
    ))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_layout(fig, "📐 ROC Curve", height=400)
