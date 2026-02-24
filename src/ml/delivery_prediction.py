"""
Late Delivery Prediction Module
Binary classifier to predict if an order will be delivered late.
Adapted for the DataCo Smart Supply Chain dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import streamlit as st
import config


class DeliveryPredictor:
    """XGBoost classifier for late delivery prediction."""

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.metrics = {}

    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Engineer features for delivery prediction."""
        data = df.copy()

        # Encode categorical variables available in DataCo
        categorical_cols = [
            "product_category", "region", "shipping_mode",
            "customer_segment", "department",
        ]
        # Only use columns that actually exist
        categorical_cols = [c for c in categorical_cols if c in data.columns]

        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                data[f"{col}_encoded"] = data[col].astype(str).map(
                    lambda x, le=le: (
                        le.transform([x])[0] if x in le.classes_ else -1
                    )
                )

        # Numerical features available in DataCo
        numerical_cols = [
            "unit_price", "quantity", "revenue", "discount_percent",
            "profit_margin", "scheduled_shipping_days",
            "order_month", "order_quarter", "order_day_of_week",
        ]
        numerical_cols = [c for c in numerical_cols if c in data.columns]

        feature_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

        # Interaction features
        if "unit_price" in data.columns and "quantity" in data.columns:
            data["price_x_quantity"] = data["unit_price"] * data["quantity"]
            feature_cols.append("price_x_quantity")

        if "discount_percent" in data.columns and "unit_price" in data.columns:
            data["discount_x_price"] = data["discount_percent"] * data["unit_price"]
            feature_cols.append("discount_x_price")

        if "scheduled_shipping_days" in data.columns and "quantity" in data.columns:
            data["scheduled_x_qty"] = data["scheduled_shipping_days"] * data["quantity"]
            feature_cols.append("scheduled_x_qty")

        # Ensure no NaNs in features
        for col in feature_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        self.feature_columns = feature_cols
        target = data["late_delivery"]

        return data[feature_cols], target

    @st.cache_resource
    def train(_self, df: pd.DataFrame):
        """Train the delivery prediction model."""
        X, y = _self._prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=y
        )

        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        _self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )

        _self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = _self.model.predict(X_test)
        y_prob = _self.model.predict_proba(X_test)[:, 1]

        _self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        return _self.metrics, y_test, y_pred, y_prob

    def predict_single(self, order_data: dict) -> dict:
        """Predict late delivery probability for a single order."""
        if self.model is None:
            return {"error": "Model not trained"}

        df = pd.DataFrame([order_data])

        # Encode categoricals
        for col in self.label_encoders:
            if col in df.columns:
                le = self.label_encoders[col]
                df[f"{col}_encoded"] = df[col].astype(str).map(
                    lambda x, le=le: (
                        le.transform([x])[0] if x in le.classes_ else -1
                    )
                )

        # Compute interaction features
        if "unit_price" in df.columns and "quantity" in df.columns:
            df["price_x_quantity"] = df["unit_price"] * df["quantity"]
        if "discount_percent" in df.columns and "unit_price" in df.columns:
            df["discount_x_price"] = df["discount_percent"] * df["unit_price"]
        if "scheduled_shipping_days" in df.columns and "quantity" in df.columns:
            df["scheduled_x_qty"] = df["scheduled_shipping_days"] * df["quantity"]

        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        prob = self.model.predict_proba(df[self.feature_columns])[0]
        prediction = self.model.predict(df[self.feature_columns])[0]

        return {
            "late_delivery_prediction": int(prediction),
            "late_probability": float(prob[1]),
            "on_time_probability": float(prob[0]),
            "risk_level": (
                "High" if prob[1] > 0.7 else
                "Medium" if prob[1] > 0.4 else "Low"
            ),
        }

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            return pd.DataFrame()
        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return importance
