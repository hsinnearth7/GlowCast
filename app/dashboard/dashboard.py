"""GlowCast Dashboard — Beauty Supply Chain Intelligence Platform.

Launch: streamlit run app/dashboard/app.py
"""

import sys
import os

# Ensure GlowCast root is on sys.path so `app.dashboard.*` imports resolve
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from app.dashboard.views import overview, forecasting, supply_chain, causal, mlops

st.set_page_config(
    page_title="GlowCast Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for premium look --
st.markdown("""
<style>
    /* ── Base dark theme ── */
    .stApp { background-color: #0e1117; color: #ecf0f1; }

    /* ── Global text: force all Streamlit text to light ── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ecf0f1 !important;
    }
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp div {
        color: #dce1e8;
    }
    .stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
    .stApp .stMarkdown strong, .stApp .stMarkdown b {
        color: #ecf0f1 !important;
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #a0aec0 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #f0f4f8 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #2d3348; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ecf0f1 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #3498db !important;
    }

    /* ── Multiselect / Select / Input labels ── */
    .stApp .stMultiSelect label, .stApp .stSelectbox label,
    .stApp .stTextInput label, .stApp .stNumberInput label,
    .stApp .stSlider label {
        color: #cbd5e1 !important;
    }
    .stApp [data-baseweb="tag"] span { color: #fff !important; }
    .stApp [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: #ecf0f1 !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(52, 152, 219, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .delta-good { color: #2ecc71 !important; }
    .delta-bad { color: #e74c3c !important; }
    .delta-neutral { color: #f39c12 !important; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ecf0f1 !important;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin: 30px 0 20px 0;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #27ae60; color: white !important; }
    .badge-red { background: #e74c3c; color: white !important; }
    .badge-yellow { background: #f39c12; color: white !important; }
    .badge-blue { background: #3498db; color: white !important; }

    /* ── Dividers ── */
    .stApp hr { border-color: #2d3348 !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, delta=None, delta_type="good", color="#3498db"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        css_class = f"delta-{delta_type}"
        delta_html = f'<div class="metric-delta {css_class}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        {delta_html}
    </div>
    """


# -- Navigation --
PAGE_OPTIONS = [
    "Executive Overview",
    "Demand & Forecasting",
    "Inventory & Supply Chain",
    "Causal & Experimentation",
    "MLOps & Quality",
]

with st.sidebar:
    st.markdown("## GlowCast")
    st.caption("Beauty Supply Chain Intelligence")
    st.divider()
    page_idx = 0
    for i, name in enumerate(PAGE_OPTIONS):
        if st.button(name, key=f"nav_{i}", use_container_width=True,
                     type="primary" if st.session_state.get("current_page", 0) == i else "secondary"):
            st.session_state["current_page"] = i
            page_idx = i
    st.divider()
    st.markdown("**Platform Stats**")
    st.markdown("- 5,000 SKUs | 12 FCs | 5 Countries")
    st.markdown("- 6 Forecast Models | 4 Uplift Learners")
    st.markdown("- 120+ Tests | 85%+ Coverage")

page_idx = st.session_state.get("current_page", 0)

# -- Page Router --
RENDERERS = [overview.render, forecasting.render, supply_chain.render, causal.render, mlops.render]
RENDERERS[page_idx](metric_card)
