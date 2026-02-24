"""
🏭 Supply Chain Intelligence Platform
Main Streamlit Application Entry Point
"""
import streamlit as st
import config

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Dark Theme ──────────────────────────────────────
st.markdown("""
<style>
    /* ─── Global Styles ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Sidebar Styling ───────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #E0E7FF;
    }

    /* ─── Metric Cards ──────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.15);
    }

    [data-testid="stMetricLabel"] {
        color: #A5B4FC !important;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: #F9FAFB !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* ─── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15,23,42,0.5);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
    }

    /* ─── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.3);
    }

    /* ─── DataFrames ────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ─── Expander ──────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(99,102,241,0.1);
        border-radius: 10px;
        font-weight: 600;
    }

    /* ─── Chat Messages ─────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: rgba(15,23,42,0.5);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* ─── Custom Badge ──────────────────────────────────── */
    .badge-success {
        background: rgba(16,185,129,0.15);
        color: #10B981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        background: rgba(245,158,11,0.15);
        color: #F59E0B;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-danger {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ─── Hero Section ──────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 50%, rgba(236,72,153,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8, #C084FC, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        position: relative;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        position: relative;
    }

    /* ─── Feature Cards ─────────────────────────────────── */
    .feature-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,27,75,0.6));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99,102,241,0.4);
        box-shadow: 0 12px 40px rgba(99,102,241,0.1);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #E0E7FF;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .feature-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ─── Info Banner ───────────────────────────────────── */
    .info-banner {
        background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(99,102,241,0.1));
        border-left: 4px solid #06B6D4;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .info-banner p {
        color: #CBD5E1;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Main Page Content ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏭 Supply Chain Intelligence Platform</div>
    <div class="hero-subtitle">
        AI-Powered Analytics • ML Forecasting • Agentic Intelligence • Real-Time Optimization
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Feature Cards ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Interactive Dashboard</div>
        <div class="feature-desc">
            Real-time KPIs, revenue trends, category breakdown,
            regional analysis with dynamic filtering.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">ML Demand Forecasting</div>
        <div class="feature-desc">
            XGBoost-powered demand predictions with time-series features,
            seasonality modeling, and confidence bands.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🚚</div>
        <div class="feature-title">Delivery Prediction</div>
        <div class="feature-desc">
            ML classifier for late delivery risk with
            feature importance and what-if scenario analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📦</div>
        <div class="feature-title">Inventory Optimization</div>
        <div class="feature-desc">
            EOQ calculations, safety stock optimization,
            reorder points, and inventory health monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🏭</div>
        <div class="feature-title">Department Intelligence</div>
        <div class="feature-desc">
            Multi-criteria department scoring, risk grading,
            radar analysis, and improvement recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI Agent (NVIDIA GPT-OSS-20B)</div>
        <div class="feature-desc">
            ReAct-style AI agent powered by NVIDIA NIM GPT-OSS-20B.
            Natural language queries with tool calling over your supply chain data.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Quick Stats ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Quick Start")

st.markdown("""
<div class="info-banner">
    <p>
        <strong>👈 Navigate using the sidebar</strong> to access the Dashboard, Forecasting, 
        Delivery Prediction, Inventory Optimization, Department Analysis, or the AI Agent.
        <br><br>
        📊 <strong>180,000+ orders</strong> • 📅 <strong>3 years of data (2015–2018)</strong> • 
        🏷️ <strong>30 product categories</strong> • 🌍 <strong>5 market regions</strong> • 
        🏭 <strong>11 departments</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Tech Stack
with st.expander("🛠️ Technology Stack", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **Machine Learning**
        - XGBoost Regression & Classification
        - Time-Series Feature Engineering
        - Scikit-learn Preprocessing
        - Statistical Optimization (EOQ, Safety Stock)
        """)
    with col_b:
        st.markdown("""
        **Agentic AI**
        - NVIDIA NIM GPT-OSS-20B (20B MoE)
        - ReAct Agent Architecture
        - 7 Supply Chain Tools
        - Conversational Memory
        """)
    with col_c:
        st.markdown("""
        **Supply Chain**
        - Demand Forecasting
        - Late Delivery Prediction
        - Inventory Optimization
        - Multi-Criteria Department Scoring
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:0.85rem;'>"
    "Built with ❤️ using Streamlit • Machine Learning • Agentic AI"
    "</p>",
    unsafe_allow_html=True,
)
