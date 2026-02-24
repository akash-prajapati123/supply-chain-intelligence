"""
Agent Tools – callable tools that the Supply Chain AI Agent can invoke.
Adapted for the DataCo Smart Supply Chain dataset.
"""
import pandas as pd
import numpy as np
import json
from typing import Optional


# ─── Tool Definitions (for the LLM function calling) ───────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_supply_chain_data",
            "description": (
                "Query the supply chain dataset. Can filter by date range, "
                "category, region, or shipping mode. Returns summary "
                "statistics of the filtered data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category to filter by",
                    },
                    "region": {
                        "type": "string",
                        "description": "Market region to filter by (Africa, Europe, LATAM, Pacific Asia, USCA)",
                    },
                    "department": {
                        "type": "string",
                        "description": "Department to filter by",
                    },
                    "metric": {
                        "type": "string",
                        "description": (
                            "Specific metric to retrieve: revenue, orders, "
                            "profit, late_delivery_rate, avg_shipping_days"
                        ),
                    },
                    "time_period": {
                        "type": "string",
                        "description": (
                            "Time period: 'last_month', 'last_quarter', "
                            "'last_year', 'all'"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_demand_forecast",
            "description": (
                "Run demand forecasting for a specific product category. "
                "Returns predicted demand for the next N days."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category to forecast",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of days to forecast (default: 30)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_supplier",
            "description": (
                "Get detailed analysis of a department's performance including "
                "delivery, profitability, reliability, and risk scores. "
                "In this dataset, departments act as supply units."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "supplier_name": {
                        "type": "string",
                        "description": "Name of the department to analyze (e.g. Fan Shop, Apparel, Golf, Technology)",
                    },
                },
                "required": ["supplier_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory_status",
            "description": (
                "Check inventory optimization metrics for a product category "
                "including EOQ, safety stock, and demand analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category to check inventory for",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delivery_risk",
            "description": (
                "Predict the risk of late delivery for specific order parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category",
                    },
                    "region": {
                        "type": "string",
                        "description": "Market region",
                    },
                    "shipping_mode": {
                        "type": "string",
                        "description": "Shipping mode (Standard Class, Second Class, First Class, Same Day)",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Order quantity",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_products",
            "description": (
                "Get the top performing products by revenue, orders, or profit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to rank by: 'revenue', 'orders', 'profit'",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top products (default: 10)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_regions",
            "description": "Compare performance metrics across all market regions.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────

def execute_tool(tool_name: str, arguments: dict, df: pd.DataFrame) -> str:
    """Route tool calls to their implementations."""
    tool_map = {
        "query_supply_chain_data": _query_data,
        "run_demand_forecast": _run_forecast,
        "analyze_supplier": _analyze_supplier,
        "check_inventory_status": _check_inventory,
        "predict_delivery_risk": _predict_delivery,
        "get_top_products": _get_top_products,
        "compare_regions": _compare_regions,
    }

    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = tool_map[tool_name](df=df, **arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _query_data(
    df: pd.DataFrame,
    category: str = None,
    region: str = None,
    department: str = None,
    metric: str = None,
    time_period: str = "all",
) -> dict:
    """Query and summarize supply chain data."""
    data = df.copy()

    # Time period filter
    if time_period == "last_month":
        cutoff = data["order_date"].max() - pd.Timedelta(days=30)
        data = data[data["order_date"] >= cutoff]
    elif time_period == "last_quarter":
        cutoff = data["order_date"].max() - pd.Timedelta(days=90)
        data = data[data["order_date"] >= cutoff]
    elif time_period == "last_year":
        cutoff = data["order_date"].max() - pd.Timedelta(days=365)
        data = data[data["order_date"] >= cutoff]

    if category:
        data = data[data["product_category"].str.contains(category, case=False)]
    if region:
        data = data[data["region"].str.contains(region, case=False)]
    if department and "department" in data.columns:
        data = data[data["department"].str.contains(department, case=False)]

    if len(data) == 0:
        return {"message": "No data found matching the filters."}

    summary = {
        "total_orders": int(len(data)),
        "total_revenue": f"${data['revenue'].sum():,.2f}",
        "total_profit": f"${data['profit'].sum():,.2f}",
        "avg_order_value": f"${data['revenue'].mean():,.2f}",
        "late_delivery_rate": f"{data['late_delivery'].mean()*100:.1f}%",
        "avg_shipping_days": f"{data['actual_shipping_days'].mean():.1f}",
        "top_category": data["product_category"].value_counts().index[0],
        "top_region": data["region"].value_counts().index[0],
        "date_range": f"{data['order_date'].min().date()} to {data['order_date'].max().date()}",
    }

    if metric:
        metric_map = {
            "revenue": ("revenue", "sum"),
            "orders": ("order_id", "count"),
            "profit": ("profit", "sum"),
            "late_delivery_rate": ("late_delivery", "mean"),
            "avg_shipping_days": ("actual_shipping_days", "mean"),
        }
        if metric in metric_map:
            col, agg = metric_map[metric]
            value = data[col].agg(agg)
            summary["requested_metric"] = {
                "metric": metric,
                "value": f"{value:,.2f}" if isinstance(value, float) else str(value),
            }

    return summary


def _run_forecast(
    df: pd.DataFrame,
    category: str = None,
    horizon_days: int = 30,
) -> dict:
    """Run demand forecast and return summary."""
    from src.ml.demand_forecasting import DemandForecaster

    forecaster = DemandForecaster()
    metrics, _, _, _ = forecaster.train(df, category)
    forecast = forecaster.forecast_future(df, horizon_days, category)

    return {
        "category": category or "All Categories",
        "forecast_horizon": f"{horizon_days} days",
        "model_metrics": {
            "MAE": f"{metrics['mae']:.1f}",
            "RMSE": f"{metrics['rmse']:.1f}",
            "R²": f"{metrics['r2']:.3f}",
        },
        "forecast_summary": {
            "avg_predicted_daily_demand": f"{forecast['predicted_demand'].mean():.0f} units",
            "peak_demand_date": str(forecast.loc[forecast['predicted_demand'].idxmax(), 'date'].date()),
            "peak_demand_value": f"{forecast['predicted_demand'].max():.0f} units",
            "total_predicted_demand": f"{forecast['predicted_demand'].sum():,.0f} units",
        },
    }


def _analyze_supplier(df: pd.DataFrame, supplier_name: str) -> dict:
    """Detailed department/supplier analysis."""
    from src.ml.supplier_scoring import SupplierScorer

    scorer = SupplierScorer()
    scores = scorer.score_suppliers(df)

    match = scores[scores["supplier"].str.contains(supplier_name, case=False)]
    if len(match) == 0:
        available = scores["supplier"].tolist()
        return {
            "error": f"Department '{supplier_name}' not found.",
            "available_suppliers": available,
        }

    supplier = match.iloc[0]
    return {
        "supplier": supplier["supplier"],
        "overall_score": f"{supplier['overall_score']:.2f}",
        "grade": str(supplier["grade"]),
        "risk_level": str(supplier["risk_level"]),
        "metrics": {
            "delivery_performance": f"{supplier['delivery_performance']*100:.1f}%",
            "profitability": f"{supplier['profitability']*100:.1f}%",
            "volume_capability": f"{supplier['volume_capability']*100:.1f}%",
            "order_reliability": f"{supplier['order_reliability']*100:.1f}%",
            "cost_efficiency": f"{supplier['cost_efficiency']*100:.1f}%",
            "product_diversity": f"{supplier['product_diversity']*100:.1f}%",
        },
        "stats": {
            "total_orders": int(supplier["total_orders"]),
            "total_revenue": f"${supplier['total_revenue']:,.2f}",
            "avg_shipping_days": f"{supplier['avg_shipping_days']:.1f} days",
            "late_delivery_rate": f"{supplier['late_delivery_rate']*100:.1f}%",
        },
    }


def _check_inventory(df: pd.DataFrame, category: str = None) -> dict:
    """Check inventory optimization status."""
    from src.ml.inventory_optimizer import InventoryOptimizer

    optimizer = InventoryOptimizer()
    results = optimizer.analyze_inventory(df)

    if category:
        match = results[results["category"].str.contains(category, case=False)]
        if len(match) == 0:
            return {"error": f"Category '{category}' not found."}
        row = match.iloc[0]
        recs = optimizer.get_recommendations(match)
        return {
            "category": row["category"],
            "avg_daily_demand": f"{row['avg_daily_demand']:.1f} units",
            "demand_variability": f"{row['demand_cv']:.1f}% CV",
            "annual_demand": f"{row['annual_demand']:,.0f} units",
            "optimal_safety_stock_95": f"{row['safety_stock_95']:.0f} units",
            "optimal_safety_stock_99": f"{row['safety_stock_99']:.0f} units",
            "eoq": f"{row['eoq']:.0f} units",
            "reorder_point": f"{row['reorder_point_95']:.0f} units",
            "late_delivery_rate": f"{row['late_delivery_rate']:.1f}%",
            "recommendations": [r["action"] for r in recs],
        }

    summary = {
        "total_categories": len(results),
        "categories": [],
    }
    for _, row in results.iterrows():
        summary["categories"].append({
            "name": row["category"],
            "eoq": f"{row['eoq']:.0f}",
            "safety_stock": f"{row['safety_stock_95']:.0f}",
            "daily_demand": f"{row['avg_daily_demand']:.1f}",
            "late_rate": f"{row['late_delivery_rate']:.1f}%",
        })
    return summary


def _predict_delivery(
    df: pd.DataFrame,
    category: str = None,
    region: str = None,
    shipping_mode: str = None,
    quantity: int = None,
) -> dict:
    """Predict delivery risk for given parameters."""
    data = df.copy()
    if category:
        cat_data = data[data["product_category"].str.contains(category, case=False)]
        if len(cat_data) > 0:
            data = cat_data
    if region:
        reg_data = data[data["region"].str.contains(region, case=False)]
        if len(reg_data) > 0:
            data = reg_data
    if shipping_mode:
        mode_data = data[data["shipping_mode"].str.contains(shipping_mode, case=False)]
        if len(mode_data) > 0:
            data = mode_data

    late_rate = data["late_delivery"].mean()
    avg_delay = data["delivery_delay_days"].mean()

    risk_level = "High" if late_rate > 0.5 else "Medium" if late_rate > 0.3 else "Low"

    return {
        "parameters": {
            "category": category or "All",
            "region": region or "All",
            "shipping_mode": shipping_mode or "All",
            "quantity": quantity or "Any",
        },
        "prediction": {
            "late_delivery_probability": f"{late_rate*100:.1f}%",
            "risk_level": risk_level,
            "avg_delay_if_late": f"{avg_delay:.1f} days",
            "matching_orders": len(data),
        },
        "recommendation": (
            "Consider using Same Day or First Class shipping."
            if risk_level == "High"
            else "Standard delivery should be acceptable."
            if risk_level == "Low"
            else "Monitor closely; consider faster shipping if timeline is critical."
        ),
    }


def _get_top_products(
    df: pd.DataFrame,
    metric: str = "revenue",
    top_n: int = 10,
) -> dict:
    """Get top products by specified metric."""
    metric_col = {
        "revenue": "revenue",
        "orders": "order_id",
        "profit": "profit",
    }.get(metric, "revenue")

    agg_func = "count" if metric == "orders" else "sum"

    top = df.groupby(["product_category", "product_name"]).agg(
        value=(metric_col, agg_func),
        avg_price=("unit_price", "mean"),
    ).reset_index().sort_values("value", ascending=False).head(top_n)

    return {
        "metric": metric,
        "top_products": [
            {
                "rank": i + 1,
                "category": row["product_category"],
                "product": row["product_name"],
                "value": f"${row['value']:,.2f}" if metric != "orders" else int(row["value"]),
                "avg_price": f"${row['avg_price']:,.2f}",
            }
            for i, (_, row) in enumerate(top.iterrows())
        ],
    }


def _compare_regions(df: pd.DataFrame) -> dict:
    """Compare all market regions."""
    regional = df.groupby("region").agg(
        total_orders=("order_id", "count"),
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
        avg_delivery_days=("actual_shipping_days", "mean"),
        late_rate=("late_delivery", "mean"),
        avg_order_value=("revenue", "mean"),
    ).reset_index()

    return {
        "regions": [
            {
                "region": row["region"],
                "orders": int(row["total_orders"]),
                "revenue": f"${row['total_revenue']:,.2f}",
                "profit": f"${row['total_profit']:,.2f}",
                "avg_delivery_days": f"{row['avg_delivery_days']:.1f}",
                "late_delivery_rate": f"{row['late_rate']*100:.1f}%",
                "avg_order_value": f"${row['avg_order_value']:,.2f}",
            }
            for _, row in regional.iterrows()
        ],
    }
