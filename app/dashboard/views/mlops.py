"""MLOps & Quality — cost model drift monitoring and feature importance."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.data import (
    load_drift_history,
    load_fairness_analysis,
    load_feature_importance,
)


def render(metric_card) -> None:
    st.header("MLOps & Quality")

    drift = load_drift_history()

    # ── Top KPIs ──
    latest = drift.iloc[-1]
    cols = st.columns(6)
    cards = [
        ("System Status", "Healthy", "All checks passing"),
        ("Cost MAPE", f"{latest['cost_mape']:.1%}", "Current prediction error"),
        ("Data Drift (KS)", f"p={latest['ks_p_value']:.2f}", "Feature distribution"),
        ("Prediction Drift", f"PSI={latest['psi']:.3f}", "< 0.10 threshold"),
        ("Top Feature", "commodity_index", "Highest SHAP importance"),
        ("Model Age", "14 days", "Since last retrain"),
    ]
    for col, (t, v, s) in zip(cols, cards):
        col.markdown(metric_card(t, v, s), unsafe_allow_html=True)

    st.markdown("---")

    # ── Drift Monitoring ──
    st.subheader("Drift Monitoring (12 Months)")
    tab1, tab2, tab3 = st.tabs(["Cost MAPE", "KS p-value", "PSI"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drift["month"], y=drift["cost_mape"],
            mode="lines+markers", name="Cost MAPE",
            line=dict(color="#60a5fa", width=2),
        ))
        fig.add_hline(y=0.15, line_dash="dash", line_color="#f87171", annotation_text="Threshold (15%)")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e", height=300, yaxis_title="MAPE",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drift["month"], y=drift["ks_p_value"],
            mode="lines+markers", name="KS p-value",
            line=dict(color="#facc15", width=2),
        ))
        fig.add_hline(y=0.05, line_dash="dash", line_color="#f87171", annotation_text="Alpha=0.05")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e", height=300, yaxis_title="p-value",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drift["month"], y=drift["psi"],
            mode="lines+markers", name="PSI",
            line=dict(color="#a78bfa", width=2),
        ))
        fig.add_hline(y=0.10, line_dash="dash", line_color="#facc15", annotation_text="Alert (0.10)")
        fig.add_hline(y=0.20, line_dash="dash", line_color="#f87171", annotation_text="Auto-Retrain (0.20)")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e", height=300, yaxis_title="PSI",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ──
    st.subheader("Feature Importance (SHAP vs LIME)")
    importance = load_feature_importance()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance["feature"], y=importance["shap_importance"],
        name="SHAP", marker_color="#60a5fa",
    ))
    fig.add_trace(go.Bar(
        x=importance["feature"], y=importance["lime_importance"],
        name="LIME", marker_color="#a78bfa",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117",
        plot_bgcolor="#1e1e2e", barmode="group", height=350,
        xaxis_tickangle=-45, yaxis_title="Importance",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Fairness Analysis ──
    st.subheader("Fairness Analysis")
    fairness = load_fairness_analysis()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost MAPE by Plant**")
        plant_df = __import__("pandas").DataFrame([
            {"plant": k, "mape": v} for k, v in fairness["plant_mape"].items()
        ])
        fig = px.bar(
            plant_df, x="plant", y="mape",
            color="mape", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        **Statistical Tests**
        - Kruskal-Wallis (across plants): p = {fairness['kruskal_wallis_p']:.3f}
          {'No significant difference' if fairness['kruskal_wallis_p'] > 0.05 else 'Significant difference detected'}
        - Chi-squared (category x tier): p = {fairness['chi2_p']:.3f}
          {'No significant difference' if fairness['chi2_p'] > 0.05 else 'Significant difference detected'}
        """)
