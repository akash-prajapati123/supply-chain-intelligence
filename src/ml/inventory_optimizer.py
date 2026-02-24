"""
Inventory Optimization Module
Statistical inventory optimization using demand patterns from the DataCo dataset.
Computes EOQ, safety stock, and reorder points from order history.
"""
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import config


class InventoryOptimizer:
    """Demand-based inventory optimization engine."""

    @staticmethod
    def calculate_eoq(
        annual_demand: float, ordering_cost: float = 50.0,
        unit_price: float = 100.0, holding_cost_rate: float = 0.20
    ) -> float:
        """Calculate Economic Order Quantity."""
        holding_cost = unit_price * holding_cost_rate
        if holding_cost <= 0 or annual_demand <= 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return round(eoq, 0)

    @staticmethod
    def calculate_safety_stock(
        avg_demand: float, demand_std: float,
        avg_lead_time: float, lead_time_std: float = 1.0,
        service_level: float = 0.95
    ) -> dict:
        """Calculate safety stock using statistical method."""
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * np.sqrt(
            (avg_lead_time * demand_std ** 2) +
            (avg_demand ** 2 * lead_time_std ** 2)
        )
        reorder_point = avg_demand * avg_lead_time + safety_stock

        return {
            "safety_stock": round(safety_stock, 0),
            "reorder_point": round(reorder_point, 0),
            "z_score": round(z_score, 2),
        }

    @st.cache_data
    def analyze_inventory(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive inventory optimization analysis per category."""
        results = []

        for category in sorted(df["product_category"].unique()):
            cat_data = df[df["product_category"] == category]

            if len(cat_data) < 10:
                continue

            # Daily demand stats
            daily_demand = cat_data.groupby(
                cat_data["order_date"].dt.date
            )["quantity"].sum()

            avg_daily_demand = daily_demand.mean()
            std_daily_demand = daily_demand.std() if len(daily_demand) > 1 else 0

            # Estimate annual demand (extrapolate from observed data range)
            date_range_days = (daily_demand.index.max() - daily_demand.index.min()).days
            if date_range_days > 0:
                annual_demand = avg_daily_demand * 365
            else:
                annual_demand = avg_daily_demand * 365

            # Use average scheduled shipping days as proxy for lead time
            avg_lead_time = cat_data["scheduled_shipping_days"].mean()
            lead_time_std = cat_data["scheduled_shipping_days"].std() if len(cat_data) > 1 else 1.0

            # Average unit price
            avg_price = cat_data["unit_price"].mean()

            # EOQ
            eoq = _self.calculate_eoq(
                annual_demand=annual_demand,
                unit_price=avg_price,
            )

            # Safety stock at 95% and 99%
            ss_95 = _self.calculate_safety_stock(
                avg_daily_demand, std_daily_demand, avg_lead_time, lead_time_std, 0.95
            )
            ss_99 = _self.calculate_safety_stock(
                avg_daily_demand, std_daily_demand, avg_lead_time, lead_time_std, 0.99
            )

            # Compute inventory health using actual late delivery data
            late_rate = cat_data["late_delivery"].mean() * 100
            # Stockout risk = proportion of times demand exceeded what could be shipped on time
            demand_variability_cv = (std_daily_demand / avg_daily_demand * 100) if avg_daily_demand > 0 else 0

            # Estimated overstock: when avg order is well below EOQ
            avg_order_qty = cat_data["quantity"].mean()
            overstock_risk = "High" if avg_order_qty > eoq * 1.5 and eoq > 0 else (
                "Medium" if avg_order_qty > eoq and eoq > 0 else "Low"
            )

            results.append({
                "category": category,
                "total_orders": len(cat_data),
                "total_revenue": round(cat_data["revenue"].sum(), 2),
                "avg_daily_demand": round(avg_daily_demand, 1),
                "demand_std": round(std_daily_demand, 1),
                "demand_cv": round(demand_variability_cv, 1),
                "annual_demand": round(annual_demand, 0),
                "avg_unit_price": round(avg_price, 2),
                "avg_lead_time": round(avg_lead_time, 1),
                "eoq": eoq,
                "safety_stock_95": ss_95["safety_stock"],
                "reorder_point_95": ss_95["reorder_point"],
                "safety_stock_99": ss_99["safety_stock"],
                "reorder_point_99": ss_99["reorder_point"],
                "late_delivery_rate": round(late_rate, 1),
                "overstock_risk": overstock_risk,
            })

        return pd.DataFrame(results)

    def get_recommendations(self, analysis_df: pd.DataFrame) -> list:
        """Generate actionable inventory recommendations."""
        recommendations = []

        for _, row in analysis_df.iterrows():
            # High demand variability
            if row["demand_cv"] > 80:
                recommendations.append({
                    "category": row["category"],
                    "priority": "High",
                    "action": f"Increase safety stock to {row['safety_stock_99']:.0f} units",
                    "reason": f"High demand variability (CV={row['demand_cv']:.0f}%). "
                              f"Current coefficient of variation exceeds 80%.",
                    "savings": None,
                })

            # High late delivery rate
            if row["late_delivery_rate"] > 60:
                recommendations.append({
                    "category": row["category"],
                    "priority": "High",
                    "action": f"Increase reorder point to {row['reorder_point_99']:.0f} units and consider faster shipping",
                    "reason": f"Late delivery rate of {row['late_delivery_rate']:.1f}% is critically high. "
                              f"Buffer stock and earlier reordering needed.",
                    "savings": None,
                })

            # EOQ optimization
            if row["eoq"] > 0 and row["avg_daily_demand"] > 0:
                optimal_orders_per_year = row["annual_demand"] / row["eoq"]
                if optimal_orders_per_year > 0:
                    order_cycle_days = 365 / optimal_orders_per_year
                    recommendations.append({
                        "category": row["category"],
                        "priority": "Medium",
                        "action": f"Order {row['eoq']:.0f} units every {order_cycle_days:.0f} days",
                        "reason": f"EOQ-based ordering minimizes total inventory cost. "
                                  f"Annual demand: {row['annual_demand']:.0f} units.",
                        "savings": None,
                    })

            # Low demand volatility — can reduce safety stock
            if row["demand_cv"] < 30 and row["late_delivery_rate"] < 40:
                recommendations.append({
                    "category": row["category"],
                    "priority": "Low",
                    "action": f"Reduce safety stock to 95% level: {row['safety_stock_95']:.0f} units",
                    "reason": f"Stable demand (CV={row['demand_cv']:.0f}%) and acceptable delivery rate "
                              f"({row['late_delivery_rate']:.1f}%) allow lower safety stock.",
                    "savings": None,
                })

        return recommendations
