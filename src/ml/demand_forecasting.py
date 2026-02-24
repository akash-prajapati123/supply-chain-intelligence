"""
Demand Forecasting Module
Uses XGBoost with time-series features for demand prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import streamlit as st
from pathlib import Path
import config


class DemandForecaster:
    """XGBoost-based demand forecasting model with time-series features."""

    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.model_path = config.MODELS_DIR / "demand_forecaster.joblib"
        self.metrics = {}

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features for forecasting."""
        features = pd.DataFrame()
        features["year"] = df["order_date"].dt.year
        features["month"] = df["order_date"].dt.month
        features["quarter"] = df["order_date"].dt.quarter
        features["day_of_week"] = df["order_date"].dt.dayofweek
        features["day_of_year"] = df["order_date"].dt.dayofyear
        features["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
        features["is_weekend"] = (df["order_date"].dt.dayofweek >= 5).astype(int)
        features["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
        features["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)
        features["is_quarter_start"] = df["order_date"].dt.is_quarter_start.astype(int)

        # Cyclical encoding for month and day
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        # Holiday season indicator
        features["is_holiday_season"] = features["month"].isin([11, 12]).astype(int)

        return features

    def _prepare_data(self, df: pd.DataFrame, category: str = None) -> tuple:
        """Prepare training data from raw supply chain data."""
        data = df.copy()
        if category:
            data = data[data["product_category"] == category]

        # Aggregate daily demand
        daily = data.groupby(data["order_date"].dt.date).agg(
            demand=("quantity", "sum"),
            revenue=("revenue", "sum"),
            num_orders=("order_id", "count"),
            avg_price=("unit_price", "mean"),
            avg_discount=("discount_percent", "mean"),
        ).reset_index()
        daily["order_date"] = pd.to_datetime(daily["order_date"])

        # Add lag features
        for lag in [1, 7, 14, 28]:
            daily[f"demand_lag_{lag}"] = daily["demand"].shift(lag)
        # Rolling features
        for window in [7, 14, 30]:
            daily[f"demand_rolling_mean_{window}"] = (
                daily["demand"].rolling(window=window, min_periods=1).mean()
            )
            daily[f"demand_rolling_std_{window}"] = (
                daily["demand"].rolling(window=window, min_periods=1).std().fillna(0)
            )

        # Time features
        time_features = self._create_time_features(daily)
        daily = pd.concat([daily, time_features], axis=1)

        # Drop NaN rows from lagging
        daily = daily.dropna()

        # Target
        target = daily["demand"]
        feature_cols = [
            c for c in daily.columns
            if c not in ["order_date", "demand"]
        ]
        features = daily[feature_cols]

        return features, target, daily

    @st.cache_resource
    def train(_self, df: pd.DataFrame, category: str = None):
        """Train the demand forecasting model."""
        features, target, _ = _self._prepare_data(df, category)
        _self.feature_columns = features.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=config.TEST_SIZE,
            shuffle=False  # Time series: don't shuffle
        )

        _self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

        _self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = _self.model.predict(X_test)
        _self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
        }

        return _self.metrics, y_test, y_pred, X_test.index

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            return pd.DataFrame()
        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return importance

    def forecast_future(
        self, df: pd.DataFrame, horizon_days: int = 90, category: str = None
    ) -> pd.DataFrame:
        """Generate future demand forecast."""
        if self.model is None:
            return pd.DataFrame()

        features, target, daily = self._prepare_data(df, category)

        last_date = daily["order_date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_days, freq="D"
        )

        # Build future features using last known values
        last_row = daily.iloc[-1]
        forecasts = []

        recent_demands = list(daily["demand"].values[-30:])

        for i, date in enumerate(future_dates):
            row = {}
            row["order_date"] = date

            # Estimate future values
            row["revenue"] = last_row["revenue"]
            row["num_orders"] = last_row["num_orders"]
            row["avg_price"] = last_row["avg_price"]
            row["avg_discount"] = last_row["avg_discount"]

            # Lag features from recent_demands
            for lag in [1, 7, 14, 28]:
                idx = len(recent_demands) - lag
                row[f"demand_lag_{lag}"] = recent_demands[idx] if idx >= 0 else recent_demands[0]

            # Rolling features
            for window in [7, 14, 30]:
                recent_window = recent_demands[-window:]
                row[f"demand_rolling_mean_{window}"] = np.mean(recent_window)
                row[f"demand_rolling_std_{window}"] = np.std(recent_window) if len(recent_window) > 1 else 0

            # Time features
            temp_df = pd.DataFrame({"order_date": [date]})
            tf = self._create_time_features(temp_df).iloc[0].to_dict()
            row.update(tf)

            feature_row = pd.DataFrame([row])
            feature_cols = [c for c in self.feature_columns if c in feature_row.columns]
            for c in self.feature_columns:
                if c not in feature_row.columns:
                    feature_row[c] = 0

            pred = self.model.predict(feature_row[self.feature_columns])[0]
            pred = max(0, pred)
            forecasts.append({"date": date, "predicted_demand": pred})
            recent_demands.append(pred)

        return pd.DataFrame(forecasts)
