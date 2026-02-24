"""
Data Processor for DataCo Smart Supply Chain Dataset.
Handles loading, column renaming, cleaning, KPI computation,
and various aggregation utilities.
"""
import pandas as pd
import numpy as np
import streamlit as st
import config


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load and preprocess the DataCo supply chain dataset."""
    df = pd.read_csv(config.RAW_DATA_PATH, encoding="latin-1")

    # Rename columns using the mapping
    df = df.rename(columns=config.COLUMN_MAP)

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=False)
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], format="mixed", dayfirst=False)

    # Derived time features
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_day_of_week"] = df["order_date"].dt.dayofweek

    # Compute delivery delay (positive = late)
    df["delivery_delay_days"] = df["actual_shipping_days"] - df["scheduled_shipping_days"]

    # Clean up: strip whitespace from categorical columns
    for col in ["product_category", "region", "shipping_mode", "customer_segment",
                 "order_status", "delivery_status", "department", "product_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure numeric types
    for col in ["revenue", "profit", "benefit", "unit_price", "quantity",
                 "discount_percent", "profit_margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical data
    df = df.dropna(subset=["order_date", "revenue"])

    return df


def get_kpi_metrics(df: pd.DataFrame) -> dict:
    """Calculate key performance indicators."""
    total_orders = len(df)
    total_revenue = df["revenue"].sum()
    total_profit = df["profit"].sum() if "profit" in df.columns else 0
    avg_order_value = df["revenue"].mean()
    on_time_rate = (1 - df["late_delivery"].mean()) * 100 if "late_delivery" in df.columns else 0
    avg_shipping_days = df["actual_shipping_days"].mean() if "actual_shipping_days" in df.columns else 0
    avg_discount = df["discount_percent"].mean() * 100 if "discount_percent" in df.columns else 0

    cancel_count = len(df[df["order_status"] == "CANCELED"]) if "order_status" in df.columns else 0
    cancellation_rate = (cancel_count / total_orders * 100) if total_orders > 0 else 0

    fraud_count = len(df[df["order_status"] == "SUSPECTED_FRAUD"]) if "order_status" in df.columns else 0
    fraud_rate = (fraud_count / total_orders * 100) if total_orders > 0 else 0

    return {
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "avg_order_value": avg_order_value,
        "on_time_rate": on_time_rate,
        "avg_shipping_days": avg_shipping_days,
        "cancellation_rate": cancellation_rate,
        "fraud_rate": fraud_rate,
        "avg_discount": avg_discount,
    }


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly revenue and order trends."""
    monthly = df.groupby(df["order_date"].dt.to_period("M")).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
        profit=("profit", "sum"),
        avg_order_value=("revenue", "mean"),
    ).reset_index()
    monthly["order_date"] = monthly["order_date"].dt.to_timestamp()
    return monthly


def get_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order breakdown by product category."""
    return df.groupby("product_category").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
        profit=("profit", "sum"),
        avg_price=("unit_price", "mean"),
        late_rate=("late_delivery", "mean"),
    ).sort_values("revenue", ascending=False).reset_index()


def get_department_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order breakdown by department."""
    if "department" not in df.columns:
        return pd.DataFrame()
    return df.groupby("department").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
        profit=("profit", "sum"),
        late_rate=("late_delivery", "mean"),
    ).sort_values("revenue", ascending=False).reset_index()


def get_region_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Performance metrics by market/region."""
    return df.groupby("region").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
        profit=("profit", "sum"),
        avg_shipping=("actual_shipping_days", "mean"),
        late_rate=("late_delivery", "mean"),
    ).sort_values("revenue", ascending=False).reset_index()


def get_shipping_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Shipping mode analysis."""
    return df.groupby("shipping_mode").agg(
        orders=("order_id", "count"),
        revenue=("revenue", "sum"),
        avg_shipping_days=("actual_shipping_days", "mean"),
        late_rate=("late_delivery", "mean"),
    ).sort_values("orders", ascending=False).reset_index()


def filter_data(df: pd.DataFrame, date_range=None, categories=None,
                regions=None, shipping_modes=None, segments=None,
                departments=None) -> pd.DataFrame:
    """Apply filters to the dataset."""
    filtered = df.copy()

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["order_date"] >= start) & (filtered["order_date"] <= end)]

    if categories:
        filtered = filtered[filtered["product_category"].isin(categories)]

    if regions:
        filtered = filtered[filtered["region"].isin(regions)]

    if shipping_modes:
        filtered = filtered[filtered["shipping_mode"].isin(shipping_modes)]

    if segments:
        filtered = filtered[filtered["customer_segment"].isin(segments)]

    if departments:
        filtered = filtered[filtered["department"].isin(departments)]

    return filtered
