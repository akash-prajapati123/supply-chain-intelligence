# 🏭 Supply Chain Intelligence Platform

An end-to-end **AI-powered Supply Chain Management Platform** that combines **Machine Learning**, **Agentic AI**, and **Supply Chain Analytics** — built with Streamlit for easy deployment.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![NVIDIA](https://img.shields.io/badge/NVIDIA-GPT--OSS--20B-76B900.svg)

## 🌟 Features

### 📊 1. Interactive Dashboard
- Real-time KPIs (Revenue, Orders, Profit, On-Time Rate)
- Monthly revenue & order volume trends
- Category & regional performance breakdowns
- Shipping mode analysis with late delivery rates
- Dynamic filtering by date, category, region, shipping mode

### 📈 2. ML Demand Forecasting
- **XGBoost** regression with time-series feature engineering
- Lag features (1, 7, 14, 28 days)
- Rolling statistics (7, 14, 30-day windows)
- Cyclical encoding for seasonality
- Configurable forecast horizon (7–180 days)
- Feature importance analysis

### 🚚 3. Late Delivery Prediction
- **XGBoost** classifier for delivery risk assessment
- Confusion matrix & ROC curve visualization
- Feature importance analysis
- **What-If Scenario Analysis** — predict risk for custom order parameters

### 📦 4. Inventory Optimization
- **Economic Order Quantity (EOQ)** calculator
- **Safety Stock** calculation with configurable service levels
- Reorder point optimization
- Overstock/understock analysis per category
- Interactive EOQ & Safety Stock calculators
- Actionable recommendations

### 🏭 5. Supplier Intelligence
- **Multi-criteria weighted scoring** (6 dimensions)
  - Delivery Performance, Quality, Reliability, Cost, Lead Time, Volume
- Risk level grading (A/B/C/D/F)
- Radar chart analysis per supplier
- Full supplier leaderboard
- Improvement suggestions

### 🤖 6. AI Agent (Agentic AI)
- **ReAct-style agent** powered by **NVIDIA NIM GPT-OSS-20B** (20B MoE model)
- OpenAI-compatible function calling via NVIDIA NIM API
- **7 supply chain tools:**
  1. Query Supply Chain Data
  2. Run Demand Forecast
  3. Analyze Supplier
  4. Check Inventory Status
  5. Predict Delivery Risk
  6. Get Top Products
  7. Compare Regions
- Conversational memory (last 10 messages)
- Rule-based fallback mode (works without API key)
- Example questions & quick actions

## 📁 Project Structure

```
ml+agent/
├── app.py                              # Main Streamlit entry point
├── config.py                           # Central configuration
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── README.md
├── data/
│   ├── generate_dataset.py             # Dataset generator (50K orders)
│   └── supply_chain_data.csv           # Generated dataset
├── src/
│   ├── data_processor.py               # Data loading & KPI computation
│   ├── ml/
│   │   ├── demand_forecasting.py       # XGBoost demand model
│   │   ├── delivery_prediction.py      # Late delivery classifier
│   │   ├── inventory_optimizer.py      # EOQ & safety stock engine
│   │   └── supplier_scoring.py         # Multi-criteria scoring
│   ├── agent/
│   │   ├── supply_chain_agent.py       # ReAct AI agent
│   │   └── tools.py                    # Agent tool definitions
│   └── visualization/
│       └── charts.py                   # Plotly chart library
├── pages/
│   ├── 1_📊_Dashboard.py
│   ├── 2_📈_Demand_Forecasting.py
│   ├── 3_🚚_Delivery_Prediction.py
│   ├── 4_📦_Inventory_Optimization.py
│   ├── 5_🏭_Supplier_Analysis.py
│   └── 6_🤖_AI_Agent.py
└── models/                             # Saved ML models (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_dataset.py
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. (Optional) Enable AI Agent
Get a **free** NVIDIA NIM API key at [build.nvidia.com](https://build.nvidia.com/), then create a `.env` file:
```
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL=openai/gpt-oss-20b
```

## 📊 Dataset

The platform uses a **realistic synthetic supply chain dataset** with:
- **50,000 orders** spanning 3 years (2022–2024)
- **34 features** including order details, shipping, pricing, supplier metrics
- **10 product categories** across **5 global regions**
- **15 suppliers** with varying quality, reliability, and lead times
- Built-in **seasonality**, **delivery delays**, and **inventory levels**

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML Models | XGBoost, Scikit-learn |
| Agentic AI | NVIDIA NIM GPT-OSS-20B (20B MoE, Apache 2.0) |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |
| Statistical Methods | SciPy, Statsmodels |

## 📝 License

This project is open source and available for personal and educational use.
