"""GlowCast Streamlit dashboard — Cost & Commercial Analytics.

Launch:  streamlit run app/dashboard/dashboard.py
"""

from __future__ import annotations

import os
import sys

# Ensure GlowCast root is on sys.path so `app.dashboard.*` imports resolve
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="GlowCast — Cost & Commercial Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 95%; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #3d3d5c; margin-bottom: 0.5rem;
    }
    .metric-card h3 { color: #8b8ba7; font-size: 0.85rem; margin: 0; }
    .metric-card h1 { color: #e0e0ff; font-size: 1.8rem; margin: 0.3rem 0; }
    .metric-card p  { color: #6b6b8d; font-size: 0.75rem; margin: 0; }
    .badge {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 8px;
        font-size: 0.75rem; font-weight: 600;
    }
    .badge-green  { background: #1a3a2a; color: #4ade80; }
    .badge-yellow { background: #3a3a1a; color: #facc15; }
    .badge-red    { background: #3a1a1a; color: #f87171; }
    .badge-blue   { background: #1a2a3a; color: #60a5fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e; border-radius: 8px;
        color: #8b8ba7; padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3d3d5c !important; color: #e0e0ff !important;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(title: str, value: str, subtitle: str = "") -> str:
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h1>{value}</h1>
        <p>{subtitle}</p>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("GlowCast")
    st.caption("Cost & Commercial Analytics")
    st.markdown("---")

    pages = [
        "Executive Overview",
        "Should-Cost & OCOGS",
        "Cost Reduction & Make-vs-Buy",
        "Causal & Experimentation",
        "MLOps & Quality",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**500 SKUs** | **12 Plants** | **5 Suppliers**")
    st.markdown("**5 Commodities** | **1,095 days**")
    st.caption("v2.0 — Cost & Commercial Analytics")

# ── Page router ──────────────────────────────────────────────────────────

if page == "Executive Overview":
    from app.dashboard.views.overview import render
    render(metric_card)

elif page == "Should-Cost & OCOGS":
    from app.dashboard.views.cost_analytics import render
    render(metric_card)

elif page == "Cost Reduction & Make-vs-Buy":
    from app.dashboard.views.cost_operations import render
    render(metric_card)

elif page == "Causal & Experimentation":
    from app.dashboard.views.causal import render
    render(metric_card)

elif page == "MLOps & Quality":
    from app.dashboard.views.mlops import render
    render(metric_card)
