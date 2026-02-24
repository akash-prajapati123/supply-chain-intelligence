"""
🤖 AI Agent Page – Conversational Supply Chain Intelligence
Powered by NVIDIA NIM API — GPT-OSS-20B (20B parameter open-source MoE model)
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data
from src.agent.supply_chain_agent import SupplyChainAgent
import config

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Supply Chain AI Agent")
st.markdown("*Powered by **NVIDIA NIM GPT-OSS-20B** — 20B parameter open-source MoE model with tool-calling*")
st.markdown("---")

# ─── Load Data ──────────────────────────────────────────────────────────────
df = load_data()

# ─── API Key Configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🟢 NVIDIA NIM Configuration")

    st.markdown("""
    <div style="
        background: rgba(118,185,0,0.1);
        border: 1px solid rgba(118,185,0,0.3);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        font-size: 0.85rem;
    ">
        <strong>🚀 GPT-OSS-20B</strong><br>
        Open-source 20B MoE model<br>
        <a href="https://build.nvidia.com/" target="_blank">Get free API key →</a>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        value=os.getenv("NVIDIA_API_KEY", ""),
        help="Get your free NVIDIA NIM API key at build.nvidia.com",
    )

    model = st.selectbox(
        "Model",
        options=[
            "openai/gpt-oss-20b",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "mistralai/mixtral-8x22b-instruct-v0.1",
        ],
        index=0,
        help="GPT-OSS-20B is the recommended model (20B params, MoE, Apache 2.0 license)",
    )

    base_url = st.text_input(
        "API Base URL",
        value=config.NVIDIA_BASE_URL,
        help="Default: https://integrate.api.nvidia.com/v1",
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state.pop("agent", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What's the total revenue?",
        "Forecast demand for Electronics",
        "Analyze the Fan Shop department",
        "Check inventory for Computers",
        "Compare market regions",
        "Top products by profit",
        "Delivery risk for Same Day shipping",
    ]
    for q in example_questions:
        if st.button(q, key=f"example_{q}", use_container_width=True):
            st.session_state["next_question"] = q

# ─── Initialize Agent ──────────────────────────────────────────────────────
agent_config_key = f"{api_key}_{model}_{base_url}"
if "agent" not in st.session_state or agent_config_key != st.session_state.get("last_agent_config", ""):
    os.environ["NVIDIA_API_KEY"] = api_key if api_key else ""
    st.session_state["agent"] = SupplyChainAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    st.session_state["last_agent_config"] = agent_config_key

agent = st.session_state["agent"]

# Status indicator
if agent.is_available:
    st.success(f"🟢 **AI Agent Active** — Using **{model}** via NVIDIA NIM with function calling")
else:
    st.info("🟡 **Rule-Based Mode** — Enter your NVIDIA API key in the sidebar for full AI agent. [Get a free key →](https://build.nvidia.com/)")

# ─── Chat History ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Handle Example Question ───────────────────────────────────────────────
next_q = st.session_state.pop("next_question", None)
if next_q:
    st.session_state["chat_history"].append({"role": "user", "content": next_q})
    with st.chat_message("user"):
        st.markdown(next_q)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing with GPT-OSS-20B..."):
            response = agent.chat(next_q, df)
        st.markdown(response)

    st.session_state["chat_history"].append({"role": "assistant", "content": response})

# ─── Chat Input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your supply chain...")

if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your supply chain data with NVIDIA GPT-OSS-20B..."):
            response = agent.chat(user_input, df)
        st.markdown(response)

    st.session_state["chat_history"].append({"role": "assistant", "content": response})

# ─── Welcome Message ───────────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(118,185,0,0.1), rgba(99,102,241,0.08));
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 30px;
    ">
        <h2 style="color: #76B900; margin-bottom: 16px;">🤖 Supply Chain AI Agent</h2>
        <p style="color: #94A3B8; font-size: 1rem; line-height: 1.6;">
            Powered by <strong style="color: #76B900;">NVIDIA NIM — GPT-OSS-20B</strong><br>
            A 20-billion parameter open-source Mixture-of-Experts model with function calling capabilities.
        </p>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 12px;">
            I can analyze your supply chain data, run ML forecasts, evaluate departments,
            optimize inventory, and predict delivery risks — all through natural language.
        </p>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 16px;">
            💬 Try asking one of the example questions in the sidebar, or type your own below!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model info
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ About GPT-OSS-20B", expanded=False):
        st.markdown("""
        | Property | Details |
        |----------|---------|
        | **Model** | GPT-OSS-20B |
        | **Architecture** | Mixture-of-Experts (MoE) |
        | **Total Parameters** | ~20 Billion |
        | **Active Parameters** | ~3.6–4B per inference |
        | **License** | Apache 2.0 (fully open) |
        | **Capabilities** | Reasoning, Tool Calling, Text Generation |
        | **API Provider** | NVIDIA NIM (OpenAI-compatible) |
        | **Free Tier** | Yes — via [build.nvidia.com](https://build.nvidia.com/) |
        """)

    # Quick Action Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    action_cols = st.columns(4)
    actions = [
        ("📊 Data Overview", "What's the overall supply chain data summary?"),
        ("📈 Run Forecast", "Forecast demand for all categories for the next 30 days"),
        ("🏭 Dept Check", "Which department has the best overall score?"),
        ("🚚 Delivery Risk", "What's the overall late delivery risk?"),
    ]
    for col, (label, question) in zip(action_cols, actions):
        with col:
            if st.button(label, key=f"action_{label}", use_container_width=True):
                st.session_state["next_question"] = question
                st.rerun()
