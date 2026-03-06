"""MLOps & Quality — Drift monitoring, feature importance, fairness, retrain triggers."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from app.dashboard.data import (
    load_drift_history, load_feature_importance, load_fairness_results,
)


def render(metric_card):
    st.title("MLOps & Quality")
    st.caption("Drift monitoring | Feature importance | Fairness analysis | Retrain triggers")

    drift = load_drift_history()
    features = load_feature_importance()
    fair = load_fairness_results()

    # ── KPI Row ──
    latest = drift.iloc[-1]
    drift_events = drift["Concept_Drift"].sum()
    data_drift_events = drift["Data_Drift"].sum()
    pred_drift_events = drift["Pred_Drift"].sum()
    current_mape = latest["MAPE"]
    is_healthy = not latest["Concept_Drift"] and not latest["Data_Drift"]

    cols = st.columns(6)
    kpis = [
        ("System Status", "Healthy" if is_healthy else "ALERT",
         "All monitors green" if is_healthy else "Drift detected",
         "good" if is_healthy else "bad", "#2ecc71" if is_healthy else "#e74c3c"),
        ("Current MAPE", f"{current_mape:.1f}%", "Latest reading", "good" if current_mape < 15 else "bad", "#3498db"),
        ("Data Drift Events", f"{data_drift_events}", f"of {len(drift)} days", "neutral", "#e67e22"),
        ("Pred Drift Events", f"{pred_drift_events}", f"PSI > 0.1", "neutral", "#9b59b6"),
        ("Concept Drift", f"{drift_events}", "MAPE > 20%", "bad" if drift_events > 3 else "good", "#e74c3c"),
        ("Top Feature", features.iloc[0]["Feature"], f'SHAP {features.iloc[0]["SHAP_mean"]:.2f}', "neutral", "#1abc9c"),
    ]
    for col, (label, value, delta, dtype, color) in zip(cols, kpis):
        col.markdown(metric_card(label, value, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("")

    # ── Drift Monitoring Timeline ──
    st.markdown('<div class="section-header">Drift Monitoring Timeline (90 Days)</div>', unsafe_allow_html=True)

    tab_mape, tab_ks, tab_psi = st.tabs(["MAPE Trend (Concept)", "KS p-value (Data)", "PSI (Prediction)"])

    with tab_mape:
        fig_mape = go.Figure()
        fig_mape.add_trace(go.Scatter(
            x=drift["Date"], y=drift["MAPE"], mode="lines",
            line=dict(color="#3498db", width=2), name="MAPE",
        ))
        # Color drift events
        concept_drift = drift[drift["Concept_Drift"]]
        fig_mape.add_trace(go.Scatter(
            x=concept_drift["Date"], y=concept_drift["MAPE"],
            mode="markers", marker=dict(color="#e74c3c", size=10),
            name="Concept Drift (MAPE>20%)",
        ))
        fig_mape.add_hline(y=20, line_dash="dash", line_color="#e74c3c",
                           annotation_text="Retrain Threshold (20%)")
        fig_mape.add_hline(y=12, line_dash="dot", line_color="#2ecc71",
                           annotation_text="Target (12%)")
        fig_mape.update_layout(
            yaxis_title="MAPE (%)", height=350,
            margin=dict(l=60, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_mape, use_container_width=True)

    with tab_ks:
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=drift["Date"], y=drift["KS_p_value"], mode="lines+markers",
            line=dict(color="#e67e22", width=1.5), marker=dict(size=4),
            name="KS p-value",
        ))
        data_drifted = drift[drift["Data_Drift"]]
        fig_ks.add_trace(go.Scatter(
            x=data_drifted["Date"], y=data_drifted["KS_p_value"],
            mode="markers", marker=dict(color="#e74c3c", size=10, symbol="x"),
            name="Data Drift (p<0.05)",
        ))
        fig_ks.add_hline(y=0.05, line_dash="dash", line_color="#e74c3c",
                         annotation_text="alpha=0.05")
        fig_ks.update_layout(
            yaxis_title="KS p-value", height=350,
            margin=dict(l=60, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    with tab_psi:
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Bar(
            x=drift["Date"], y=drift["PSI"],
            marker_color=[
                "#e74c3c" if p > 0.2 else "#e67e22" if p > 0.1 else "#2ecc71"
                for p in drift["PSI"]
            ],
            name="PSI",
        ))
        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="#e67e22",
                          annotation_text="Alert (0.1)")
        fig_psi.add_hline(y=0.2, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Auto-Retrain (0.2)")
        fig_psi.update_layout(
            yaxis_title="PSI", height=350,
            margin=dict(l=60, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    # ── Feature Importance + Fairness ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Feature Importance (SHAP vs LIME)</div>', unsafe_allow_html=True)
        cat_colors = {"Demand": "#3498db", "Social": "#e67e22", "Climate": "#2ecc71", "Calendar": "#9b59b6"}
        sorted_feat = features.sort_values("SHAP_mean", ascending=True)

        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            y=sorted_feat["Feature"], x=sorted_feat["SHAP_mean"],
            name="SHAP", orientation="h",
            marker_color=[cat_colors.get(c, "#95a5a6") for c in sorted_feat["Category"]],
            text=[f"{v:.2f}" for v in sorted_feat["SHAP_mean"]], textposition="outside",
        ))
        fig_feat.add_trace(go.Scatter(
            y=sorted_feat["Feature"], x=sorted_feat["LIME_mean"],
            mode="markers", name="LIME",
            marker=dict(color="#f39c12", size=10, symbol="diamond"),
        ))
        fig_feat.update_layout(
            xaxis_title="Mean |Importance|", height=420,
            margin=dict(l=140, r=40, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"), xaxis=dict(gridcolor="#2d3348"),
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Fairness Analysis (MAPE by Segment)</div>', unsafe_allow_html=True)
        seg_fair = fair["segments"]
        fig_fair = go.Figure()
        for texture in ["Lightweight", "Rich"]:
            tdf = seg_fair[seg_fair["Texture"] == texture]
            fig_fair.add_trace(go.Bar(
                x=tdf["Concern"], y=tdf["MAPE"],
                name=texture,
                error_y=dict(type="data", symmetric=False,
                             array=(tdf["CI_hi"] - tdf["MAPE"]).tolist(),
                             arrayminus=(tdf["MAPE"] - tdf["CI_lo"]).tolist()),
                marker_color="#3498db" if texture == "Lightweight" else "#e67e22",
                text=[f"{v:.1f}%" for v in tdf["MAPE"]], textposition="outside",
            ))
        fig_fair.update_layout(
            barmode="group", yaxis_title="MAPE (%)", height=350,
            margin=dict(l=60, r=20, t=10, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"), yaxis=dict(gridcolor="#2d3348", range=[0, 20]),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_fair, use_container_width=True)

        # Statistical tests
        kw = fair["kruskal_wallis"]
        chi = fair["chi_squared"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            **Kruskal-Wallis (FC parity)**
            - H = {kw['H']}, p = {kw['p']}
            - {'<span class="badge badge-green">PASS</span>' if kw['p'] > 0.05 else '<span class="badge badge-red">FAIL</span>'}
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            **Chi-Squared (Category bias)**
            - X2 = {chi['chi2']}, p = {chi['p']}
            - {'<span class="badge badge-green">PASS</span>' if chi['p'] > 0.05 else '<span class="badge badge-red">FAIL</span>'}
            """, unsafe_allow_html=True)

    # ── Retrain Decision Flow ──
    st.markdown('<div class="section-header">Automated Retrain Decision Flow</div>', unsafe_allow_html=True)

    fig_flow = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="#2d3348", width=1),
            label=["Daily MAPE", "KS Test", "PSI Check",
                   "Concept Drift?", "Data Drift?", "Pred Drift?",
                   "OK (Normal)", "ALERT", "AUTO_RETRAIN",
                   "Retrain Pipeline", "Deploy if Improved"],
            color=["#3498db", "#e67e22", "#9b59b6",
                   "#f39c12", "#f39c12", "#f39c12",
                   "#2ecc71", "#e67e22", "#e74c3c",
                   "#8e44ad", "#27ae60"],
        ),
        link=dict(
            source=[0, 1, 2, 3, 3, 4, 4, 5, 5, 7, 8, 9],
            target=[3, 4, 5, 6, 8, 6, 7, 6, 7, 9, 9, 10],
            value= [20, 20, 20, 15, 5, 15, 5, 15, 5, 5, 5, 10],
            color=["rgba(52,152,219,0.3)"] * 3 +
                  ["rgba(46,204,113,0.3)", "rgba(231,76,60,0.3)"] * 3 +
                  ["rgba(230,126,34,0.3)", "rgba(231,76,60,0.3)", "rgba(142,68,173,0.3)"],
        ),
    ))
    fig_flow.update_layout(
        height=300, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1", size=12),
    )
    st.plotly_chart(fig_flow, use_container_width=True)
