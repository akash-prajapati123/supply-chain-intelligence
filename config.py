"""
Configuration settings for the Supply Chain Intelligence Platform.
Uses the real DataCo Smart Supply Chain dataset (180K+ orders).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = DATA_DIR / "DataCoSupplyChainDataset.csv"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─── NVIDIA NIM API (for Agentic AI — GPT-OSS-20B) ─────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

# ─── App Settings ────────────────────────────────────────────────────────────
APP_TITLE = "🏭 Supply Chain Intelligence Platform"
APP_ICON = "🏭"
APP_LAYOUT = "wide"

# ─── ML Model Parameters ────────────────────────────────────────────────────
FORECAST_HORIZON_DAYS = 90
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Column Mapping (DataCo → Internal Names) ──────────────────────────────
# Maps the original DataCo column names to our standardized internal names.
COLUMN_MAP = {
    "order date (DateOrders)": "order_date",
    "shipping date (DateOrders)": "shipping_date",
    "Category Name": "product_category",
    "Product Name": "product_name",
    "Market": "region",
    "Order Region": "sub_region",
    "Customer Segment": "customer_segment",
    "Shipping Mode": "shipping_mode",
    "Order Status": "order_status",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery",
    "Order Item Quantity": "quantity",
    "Product Price": "unit_price",
    "Sales": "revenue",
    "Order Profit Per Order": "profit",
    "Benefit per order": "benefit",
    "Order Item Total": "total_price",
    "Days for shipping (real)": "actual_shipping_days",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Order Item Discount Rate": "discount_percent",
    "Order Item Profit Ratio": "profit_margin",
    "Order Id": "order_id",
    "Department Name": "department",
    "Customer City": "customer_city",
    "Customer Country": "customer_country",
    "Order Country": "order_country",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Type": "payment_type",
}

# ─── Domain Constants (from the real dataset) ──────────────────────────────
PRODUCT_CATEGORIES = [
    "Accessories", "Baby", "Basketball", "Books", "Camping & Hiking",
    "Cardio Equipment", "Children's Clothing", "Cleats", "Computers",
    "Consumer Electronics", "Electronics", "Fishing", "Fitness Accessories",
    "Garden", "Golf Apparel", "Golf Bags & Carts", "Health and Beauty",
    "Hockey", "Indoor/Outdoor Games", "Men's Clothing", "Men's Footwear",
    "Music", "Pet Supplies", "Soccer", "Sporting Goods", "Strength Training",
    "Tennis & Racquet", "Toys", "Video Games", "Women's Apparel",
    "Women's Clothing",
]

REGIONS = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]

SHIPPING_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]

ORDER_STATUSES = [
    "COMPLETE", "CLOSED", "PENDING", "PENDING_PAYMENT",
    "PROCESSING", "ON_HOLD", "CANCELED", "SUSPECTED_FRAUD", "PAYMENT_REVIEW",
]

DEPARTMENTS = [
    "Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors",
    "Fitness", "Book Shop", "Discs Shop", "Technology",
    "Pet Shop", "Health and Beauty",
]
