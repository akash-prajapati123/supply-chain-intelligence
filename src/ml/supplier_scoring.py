"""
Department / Supplier Scoring Module
Multi-criteria performance assessment by department.
The DataCo dataset doesn't have explicit supplier IDs, so we use
Department Name as the scoring entity (each department = a supply unit).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import config


class SupplierScorer:
    """Multi-dimensional department/supplier performance scoring system."""

    WEIGHTS = {
        "delivery_performance": 0.30,
        "profitability": 0.20,
        "volume_capability": 0.15,
        "order_reliability": 0.15,
        "cost_efficiency": 0.10,
        "product_diversity": 0.10,
    }

    @st.cache_data
    def score_suppliers(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive department/supplier scores."""
        group_col = "department" if "department" in df.columns else "product_category"

        supplier_data = df.groupby(group_col).agg(
            total_orders=("order_id", "count"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            avg_shipping_days=("actual_shipping_days", "mean"),
            avg_scheduled_days=("scheduled_shipping_days", "mean"),
            late_delivery_rate=("late_delivery", "mean"),
            avg_delay=("delivery_delay_days", "mean"),
            total_quantity=("quantity", "sum"),
            num_categories=("product_category", "nunique"),
            avg_discount=("discount_percent", "mean"),
        ).reset_index()
        supplier_data = supplier_data.rename(columns={group_col: "supplier"})

        scaler = MinMaxScaler()

        # 1. Delivery performance (lower late rate = better)
        supplier_data["delivery_performance"] = 1 - supplier_data["late_delivery_rate"]

        # 2. Profitability (higher avg profit margin = better)
        # Normalize profit margin to 0-1 scale
        profit_vals = supplier_data[["avg_profit_margin"]].copy()
        profit_vals = profit_vals.fillna(0)
        if profit_vals["avg_profit_margin"].std() > 0:
            supplier_data["profitability"] = scaler.fit_transform(profit_vals).flatten()
        else:
            supplier_data["profitability"] = 0.5

        # 3. Volume capability (higher total quantity = better)
        supplier_data["volume_capability"] = scaler.fit_transform(
            supplier_data[["total_quantity"]]
        ).flatten()

        # 4. Order reliability (percentage completed / not cancelled)
        # Use inverse of late rate as a proxy
        supplier_data["order_reliability"] = np.clip(
            1 - (supplier_data["avg_delay"].clip(lower=0) /
                 supplier_data["avg_scheduled_days"].clip(lower=1)),
            0, 1
        )

        # 5. Cost efficiency (lower avg discount = more efficient pricing)
        if supplier_data["avg_discount"].std() > 0:
            supplier_data["cost_efficiency"] = 1 - scaler.fit_transform(
                supplier_data[["avg_discount"]]
            ).flatten()
        else:
            supplier_data["cost_efficiency"] = 0.5

        # 6. Product diversity (more categories = more capable)
        supplier_data["product_diversity"] = scaler.fit_transform(
            supplier_data[["num_categories"]]
        ).flatten()

        # Overall weighted score
        score_cols = list(_self.WEIGHTS.keys())
        supplier_data["overall_score"] = sum(
            supplier_data[metric] * weight
            for metric, weight in _self.WEIGHTS.items()
        )

        # Risk level and grade
        supplier_data["risk_level"] = pd.cut(
            supplier_data["overall_score"],
            bins=[0, 0.3, 0.5, 0.65, 0.8, 1.0],
            labels=["Critical", "High", "Medium", "Low", "Very Low"],
        )
        supplier_data["grade"] = pd.cut(
            supplier_data["overall_score"],
            bins=[0, 0.3, 0.5, 0.65, 0.8, 1.0],
            labels=["F", "D", "C", "B", "A"],
        )

        return supplier_data.sort_values("overall_score", ascending=False)

    def get_risk_summary(self, scored_df: pd.DataFrame) -> dict:
        """Get summary of risk distribution."""
        risk_counts = scored_df["risk_level"].value_counts().to_dict()
        return {
            "total_entities": len(scored_df),
            "avg_score": round(scored_df["overall_score"].mean(), 3),
            "best": scored_df.iloc[0]["supplier"] if len(scored_df) > 0 else "N/A",
            "worst": scored_df.iloc[-1]["supplier"] if len(scored_df) > 0 else "N/A",
            "risk_distribution": risk_counts,
        }

    def get_improvement_suggestions(self, scored_df: pd.DataFrame) -> list:
        """Generate actionable improvement suggestions."""
        suggestions = []
        for _, row in scored_df.iterrows():
            weak_areas = []
            for metric, weight in self.WEIGHTS.items():
                if row[metric] < 0.5:
                    weak_areas.append((metric, row[metric], weight))

            if weak_areas:
                weak_areas.sort(key=lambda x: x[1])
                top_weak = weak_areas[0]
                action = {
                    "delivery_performance": f"Late rate is {row['late_delivery_rate']*100:.1f}%. Consider faster shipping modes or better scheduling.",
                    "profitability": f"Profit margin is low ({row['avg_profit_margin']*100:.1f}%). Review pricing or reduce discounts.",
                    "volume_capability": "Low order volume. Explore demand growth or consolidate orders.",
                    "order_reliability": f"Average delay is {row['avg_delay']:.1f} days. Improve scheduling accuracy.",
                    "cost_efficiency": f"Average discount rate is {row['avg_discount']*100:.1f}%. Reduce unnecessary discounts.",
                    "product_diversity": f"Only {row['num_categories']} categories. Expand product range.",
                }.get(top_weak[0], "Review overall performance.")

                suggestions.append({
                    "supplier": row["supplier"],
                    "score": round(row["overall_score"], 3),
                    "grade": row["grade"],
                    "risk_level": row["risk_level"],
                    "weakest_area": top_weak[0].replace("_", " ").title(),
                    "suggestion": action,
                })

        return suggestions
