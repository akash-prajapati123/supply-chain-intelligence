"""
Supply Chain AI Agent
ReAct-style agent using NVIDIA NIM API (GPT-OSS-20B) with OpenAI-compatible function calling.
Can query data, run ML models, and provide actionable insights.
"""
import json
import pandas as pd
from openai import OpenAI
from typing import Generator
import config
from src.agent.tools import TOOL_DEFINITIONS, execute_tool


SYSTEM_PROMPT = """You are an expert Supply Chain Intelligence Agent. You have access to the DataCo Smart Supply Chain dataset with 180,000+ orders spanning 3 years (2015-2018). You can analyze data, run forecasts, evaluate department performance, check inventory, and predict delivery risks.

Your capabilities:
1. **Data Querying**: Filter and analyze supply chain data by category, market region, department, time period
2. **Demand Forecasting**: Run ML-based demand forecasts for any product category
3. **Department Analysis**: Score and evaluate department performance across multiple dimensions
4. **Inventory Optimization**: Compute EOQ, safety stock, and reorder points from demand data
5. **Delivery Risk Prediction**: Predict likelihood of late delivery for specific order parameters
6. **Product Analytics**: Find top-performing products by revenue, orders, or profit
7. **Regional Comparison**: Compare performance across all market regions

Guidelines:
- Always use available tools to ground your answers in data
- Provide specific numbers and actionable recommendations
- When comparing, use relative terms (X% better/worse)
- Highlight risks and opportunities
- Format responses clearly with bullet points and sections
- If asked about something outside your data, say so clearly
- Be proactive in suggesting related insights

Available product categories: Accessories, Baby, Basketball, Books, Camping & Hiking, Cardio Equipment, Children's Clothing, Cleats, Computers, Consumer Electronics, Electronics, Fishing, Fitness Accessories, Garden, Golf Apparel, Health and Beauty, Hockey, Indoor/Outdoor Games, Men's Clothing, Men's Footwear, Music, Pet Supplies, Soccer, Sporting Goods, Strength Training, Tennis & Racquet, Toys, Video Games, Women's Apparel, Women's Clothing

Available market regions: Africa, Europe, LATAM, Pacific Asia, USCA

Available shipping modes: Standard Class, Second Class, First Class, Same Day

Available departments: Fan Shop, Apparel, Golf, Footwear, Outdoors, Fitness, Book Shop, Discs Shop, Technology, Pet Shop, Health and Beauty
"""


class SupplyChainAgent:
    """ReAct-style AI agent with supply chain tools, powered by NVIDIA NIM GPT-OSS-20B."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or config.NVIDIA_API_KEY
        self.base_url = base_url or config.NVIDIA_BASE_URL
        self.model = model or config.NVIDIA_MODEL
        self.client = None
        self.conversation_history = []
        self.max_iterations = 5  # Max tool-calling loops

        if self.api_key and self.api_key not in ("", "nvapi-your-nvidia-api-key-here"):
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except Exception:
                self.client = None

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def chat(self, user_message: str, df: pd.DataFrame) -> str:
        """Process a user message and return agent response."""
        if not self.is_available:
            return self._fallback_response(user_message, df)

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + self.conversation_history[-10:]  # Keep last 10 messages

        for iteration in range(self.max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as e:
                error_msg = str(e)
                # Helpful error messages for common issues
                if "401" in error_msg or "Unauthorized" in error_msg:
                    return "⚠️ **Invalid NVIDIA API Key.** Please get a free key at [build.nvidia.com](https://build.nvidia.com/) and enter it in the sidebar."
                elif "429" in error_msg:
                    return "⚠️ **Rate limit exceeded.** Please wait a moment and try again."
                elif "404" in error_msg:
                    return f"⚠️ **Model not found:** `{self.model}`. Check the model name in NVIDIA NIM catalog."
                return f"⚠️ **NVIDIA NIM API Error:** {error_msg}"

            message = response.choices[0].message

            # If there are tool calls, execute them
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    result = execute_tool(tool_name, arguments, df)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                # No tool calls — we have the final answer
                assistant_message = message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message,
                })
                return assistant_message

        return "⚠️ Reached maximum reasoning iterations. Please try a simpler question."

    def _fallback_response(self, user_message: str, df: pd.DataFrame) -> str:
        """Rule-based fallback when no API key is available."""
        msg = user_message.lower()

        # Route to appropriate tool
        if any(w in msg for w in ["revenue", "sales", "order", "how many", "total"]):
            result = execute_tool("query_supply_chain_data", {
                "time_period": "last_year" if "year" in msg else "all"
            }, df)
            data = json.loads(result)
            return self._format_query_response(data)

        elif any(w in msg for w in ["forecast", "predict demand", "future demand"]):
            category = self._extract_category(msg)
            result = execute_tool("run_demand_forecast", {
                "category": category, "horizon_days": 30
            }, df)
            data = json.loads(result)
            return self._format_forecast_response(data)

        elif any(w in msg for w in ["supplier", "vendor", "department"]):
            supplier = self._extract_department(msg)
            if supplier:
                result = execute_tool("analyze_supplier", {
                    "supplier_name": supplier
                }, df)
                data = json.loads(result)
                return self._format_supplier_response(data)
            else:
                result = execute_tool("query_supply_chain_data", {}, df)
                data = json.loads(result)
                return self._format_query_response(data)

        elif any(w in msg for w in ["inventory", "stock", "reorder"]):
            category = self._extract_category(msg)
            result = execute_tool("check_inventory_status", {
                "category": category
            }, df)
            data = json.loads(result)
            return self._format_inventory_response(data)

        elif any(w in msg for w in ["delivery", "late", "delay", "shipping"]):
            result = execute_tool("predict_delivery_risk", {
                "category": self._extract_category(msg),
                "shipping_mode": self._extract_shipping(msg),
            }, df)
            data = json.loads(result)
            return self._format_delivery_response(data)

        elif any(w in msg for w in ["top", "best", "worst", "ranking"]):
            metric = "profit" if "profit" in msg else "revenue"
            result = execute_tool("get_top_products", {
                "metric": metric, "top_n": 10
            }, df)
            data = json.loads(result)
            return self._format_top_products_response(data)

        elif any(w in msg for w in ["region", "compare", "geography", "market"]):
            result = execute_tool("compare_regions", {}, df)
            data = json.loads(result)
            return self._format_region_response(data)

        else:
            return self._general_help()

    def _extract_category(self, msg: str) -> str:
        for cat in config.PRODUCT_CATEGORIES:
            if cat.lower() in msg:
                return cat
        return None

    def _extract_department(self, msg: str) -> str:
        for dept in config.DEPARTMENTS:
            if dept.lower() in msg:
                return dept
        return None

    def _extract_shipping(self, msg: str) -> str:
        for mode in config.SHIPPING_MODES:
            if mode.lower() in msg:
                return mode
        return None

    def _format_query_response(self, data: dict) -> str:
        if "error" in data:
            return f"❌ {data['error']}"
        return f"""📊 **Supply Chain Data Summary**

| Metric | Value |
|--------|-------|
| Total Orders | {data.get('total_orders', 'N/A'):,} |
| Total Revenue | {data.get('total_revenue', 'N/A')} |
| Total Profit | {data.get('total_profit', 'N/A')} |
| Avg Order Value | {data.get('avg_order_value', 'N/A')} |
| Late Delivery Rate | {data.get('late_delivery_rate', 'N/A')} |
| Avg Shipping Days | {data.get('avg_shipping_days', 'N/A')} |
| Top Category | {data.get('top_category', 'N/A')} |
| Top Region | {data.get('top_region', 'N/A')} |
| Date Range | {data.get('date_range', 'N/A')} |

💡 *Try asking about specific categories, departments, or market regions for detailed analysis!*"""

    def _format_forecast_response(self, data: dict) -> str:
        if "error" in data:
            return f"❌ {data['error']}"
        fs = data.get("forecast_summary", {})
        mm = data.get("model_metrics", {})
        return f"""🔮 **Demand Forecast: {data.get('category', 'All')}**

**Forecast Period:** {data.get('forecast_horizon', '30 days')}

📈 **Predictions:**
- Average Daily Demand: {fs.get('avg_predicted_daily_demand', 'N/A')}
- Peak Demand Date: {fs.get('peak_demand_date', 'N/A')}
- Peak Demand: {fs.get('peak_demand_value', 'N/A')}
- Total Predicted Demand: {fs.get('total_predicted_demand', 'N/A')}

🎯 **Model Accuracy:**
- MAE: {mm.get('MAE', 'N/A')}
- RMSE: {mm.get('RMSE', 'N/A')}
- R²: {mm.get('R²', 'N/A')}"""

    def _format_supplier_response(self, data: dict) -> str:
        if "error" in data:
            avail = data.get("available_suppliers", [])
            return f"❌ {data['error']}\n\n**Available departments:** {', '.join(avail)}"
        metrics = data.get("metrics", {})
        stats = data.get("stats", {})
        return f"""🏭 **Department Analysis: {data.get('supplier', '')}**

**Overall Score:** {data.get('overall_score', 'N/A')} | **Grade:** {data.get('grade', 'N/A')} | **Risk:** {data.get('risk_level', 'N/A')}

📊 **Performance Metrics:**
- Delivery Performance: {metrics.get('delivery_performance', 'N/A')}
- Profitability: {metrics.get('profitability', 'N/A')}
- Volume Capability: {metrics.get('volume_capability', 'N/A')}
- Order Reliability: {metrics.get('order_reliability', 'N/A')}
- Cost Efficiency: {metrics.get('cost_efficiency', 'N/A')}
- Product Diversity: {metrics.get('product_diversity', 'N/A')}

📦 **Order Statistics:**
- Total Orders: {stats.get('total_orders', 'N/A')}
- Total Revenue: {stats.get('total_revenue', 'N/A')}
- Avg Shipping Days: {stats.get('avg_shipping_days', 'N/A')}
- Late Delivery Rate: {stats.get('late_delivery_rate', 'N/A')}"""

    def _format_inventory_response(self, data: dict) -> str:
        if "error" in data:
            return f"❌ {data['error']}"
        if "categories" in data:
            rows = "\n".join([
                f"| {c['name']} | {c['eoq']} | {c['safety_stock']} | {c['daily_demand']} | {c['late_rate']} |"
                for c in data["categories"]
            ])
            return f"""📦 **Inventory Optimization Overview**

| Category | EOQ | Safety Stock | Daily Demand | Late Rate |
|----------|-----|-------------|-------------|-----------|
{rows}"""
        recs = data.get("recommendations", [])
        rec_text = "\n".join([f"- {r}" for r in recs]) if recs else "No critical issues detected."
        return f"""📦 **Inventory: {data.get('category', '')}**

| Metric | Value |
|--------|-------|
| Avg Daily Demand | {data.get('avg_daily_demand', 'N/A')} |
| Demand Variability | {data.get('demand_variability', 'N/A')} |
| Annual Demand | {data.get('annual_demand', 'N/A')} |
| Safety Stock (95%) | {data.get('optimal_safety_stock_95', 'N/A')} |
| Safety Stock (99%) | {data.get('optimal_safety_stock_99', 'N/A')} |
| EOQ | {data.get('eoq', 'N/A')} |
| Reorder Point | {data.get('reorder_point', 'N/A')} |
| Late Delivery Rate | {data.get('late_delivery_rate', 'N/A')} |

🔔 **Recommendations:**
{rec_text}"""

    def _format_delivery_response(self, data: dict) -> str:
        pred = data.get("prediction", {})
        params = data.get("parameters", {})
        return f"""🚚 **Delivery Risk Prediction**

**Parameters:** Category={params.get('category','All')}, Region={params.get('region','All')}, Mode={params.get('shipping_mode','All')}

| Metric | Value |
|--------|-------|
| Late Delivery Probability | {pred.get('late_delivery_probability', 'N/A')} |
| Risk Level | {pred.get('risk_level', 'N/A')} |
| Avg Delay if Late | {pred.get('avg_delay_if_late', 'N/A')} |
| Based on | {pred.get('matching_orders', 'N/A')} similar orders |

💡 **Recommendation:** {data.get('recommendation', 'N/A')}"""

    def _format_top_products_response(self, data: dict) -> str:
        products = data.get("top_products", [])
        rows = "\n".join([
            f"| {p['rank']} | {p['category']} | {p['product']} | {p['value']} | {p['avg_price']} |"
            for p in products
        ])
        return f"""🏆 **Top Products by {data.get('metric', 'revenue').title()}**

| Rank | Category | Product | Value | Avg Price |
|------|----------|---------|-------|-----------|
{rows}"""

    def _format_region_response(self, data: dict) -> str:
        regions = data.get("regions", [])
        rows = "\n".join([
            f"| {r['region']} | {r['orders']} | {r['revenue']} | {r['late_delivery_rate']} | {r['avg_delivery_days']} |"
            for r in regions
        ])
        return f"""🌍 **Market Region Performance Comparison**

| Region | Orders | Revenue | Late Rate | Avg Delivery |
|--------|--------|---------|-----------|-------------|
{rows}"""

    def _general_help(self) -> str:
        return """👋 **Welcome to the Supply Chain Intelligence Agent!**

I can help you with:

🔍 **Data Analysis** — *"What's the total revenue for Electronics?"*
📈 **Demand Forecasting** — *"Forecast demand for Sporting Goods"*
🏭 **Department Analysis** — *"Analyze the Fan Shop department"*
📦 **Inventory Health** — *"Check inventory for Computers"*
🚚 **Delivery Prediction** — *"What's the delivery risk for Same Day shipping?"*
🏆 **Product Rankings** — *"Show top products by profit"*
🌍 **Regional Comparison** — *"Compare performance across market regions"*

Just ask me anything about your supply chain! 💬"""
