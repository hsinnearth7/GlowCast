"""Cost Reduction & Make-vs-Buy — operations and optimization."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.data import (
    load_cost_anomalies,
    load_cost_reduction_tracking,
    load_make_vs_buy_comparison,
    load_supplier_performance,
)


def render(metric_card) -> None:
    st.header("Cost Reduction & Make-vs-Buy")

    # ── KPI Cards ──
    cols = st.columns(6)
    cards = [
        ("Actions Tracked", "78", "Across 8 action types"),
        ("Avg Savings", "7.4%", "Completed actions"),
        ("Realization Rate", "82%", "Actual vs Projected"),
        ("Make Recommended", "12", "Of 20 analyzed SKUs"),
        ("Buy Recommended", "6", "Cost advantage > 5%"),
        ("Cost Anomalies", "8", "Z-score > 2.5"),
    ]
    for col, (t, v, s) in zip(cols, cards):
        col.markdown(metric_card(t, v, s), unsafe_allow_html=True)

    st.markdown("---")

    # ── Cost Reduction Tracking ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost Reduction by Action Type")
        tracking = load_cost_reduction_tracking()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tracking["action_type"], y=tracking["avg_projected_pct"],
            name="Projected", marker_color="#60a5fa",
        ))
        fig.add_trace(go.Bar(
            x=tracking["action_type"], y=tracking["avg_actual_pct"],
            name="Actual", marker_color="#4ade80",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            barmode="group",
            height=350,
            xaxis_tickangle=-45,
            yaxis_title="Savings %",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Make-vs-Buy Analysis")
        mvb = load_make_vs_buy_comparison()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mvb["sku_id"], y=mvb["make_cost"],
            name="Make Cost", marker_color="#60a5fa",
        ))
        fig.add_trace(go.Bar(
            x=mvb["sku_id"], y=mvb["buy_cost"],
            name="Buy Cost", marker_color="#f87171",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            barmode="group",
            height=350,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Supplier Performance ──
    st.subheader("Supplier Performance Matrix")
    suppliers = load_supplier_performance()
    fig = go.Figure(data=go.Scatterpolar(
        r=[suppliers.iloc[0]["quality_score"], suppliers.iloc[0]["on_time_pct"],
           1 - suppliers.iloc[0]["price_premium"], 1 - suppliers.iloc[0]["lead_time_days"] / 30],
        theta=["Quality", "On-Time", "Cost", "Speed"],
        fill="toself", name=suppliers.iloc[0]["supplier"],
    ))
    for i in range(1, len(suppliers)):
        fig.add_trace(go.Scatterpolar(
            r=[suppliers.iloc[i]["quality_score"], suppliers.iloc[i]["on_time_pct"],
               1 - suppliers.iloc[i]["price_premium"], 1 - suppliers.iloc[i]["lead_time_days"] / 30],
            theta=["Quality", "On-Time", "Cost", "Speed"],
            fill="toself", name=suppliers.iloc[i]["supplier"],
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        polar=dict(bgcolor="#1e1e2e"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cost Anomalies ──
    st.subheader("Cost Anomaly Detection")
    anomalies = load_cost_anomalies()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomalies["date"], y=anomalies["z_score"],
        mode="lines", name="Z-Score",
        line=dict(color="#60a5fa", width=1),
    ))
    fig.add_hline(y=2.5, line_dash="dash", line_color="#f87171", annotation_text="Alert Threshold")
    fig.add_hline(y=-2.5, line_dash="dash", line_color="#f87171")
    anomaly_pts = anomalies[anomalies["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomaly_pts["date"], y=anomaly_pts["z_score"],
        mode="markers", name="Anomaly",
        marker=dict(color="#f87171", size=8),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e1e2e",
        height=300,
        yaxis_title="Z-Score",
    )
    st.plotly_chart(fig, use_container_width=True)
