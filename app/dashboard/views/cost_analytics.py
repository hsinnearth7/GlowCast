"""Should-Cost & OCOGS — cost decomposition and tracking."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.data import (
    load_commodity_price_index,
    load_cost_variance_heatmap,
    load_ocogs_trend,
    load_should_cost_breakdown,
)


def render(metric_card) -> None:
    st.header("Should-Cost & OCOGS")

    # ── KPI Cards ──
    cols = st.columns(6)
    cards = [
        ("Avg Should-Cost", "$46.20", "Across 500 SKUs"),
        ("Avg Actual Cost", "$52.00", "Current landed cost"),
        ("Cost Gap", "11.2%", "Should vs Actual"),
        ("OCOGS Trend", "-4.8%", "12-month change"),
        ("Top Gap Category", "Components", "Highest variance"),
        ("Commodities Tracked", "5", "Daily price index"),
    ]
    for col, (t, v, s) in zip(cols, cards):
        col.markdown(metric_card(t, v, s), unsafe_allow_html=True)

    st.markdown("---")

    # ── Should-Cost Waterfall ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Should-Cost Decomposition")
        breakdown = load_should_cost_breakdown()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=breakdown["element"], y=breakdown["should_cost"],
            name="Should-Cost", marker_color="#60a5fa",
        ))
        fig.add_trace(go.Bar(
            x=breakdown["element"], y=breakdown["actual_cost"],
            name="Actual Cost", marker_color="#f87171",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            barmode="group",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("OCOGS Monthly Trend")
        ocogs = load_ocogs_trend()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ocogs["month"], y=ocogs["avg_unit_cost"],
            mode="lines+markers", name="Actual",
            line=dict(color="#60a5fa", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ocogs["month"], y=ocogs["target_cost"],
            mode="lines", name="Target",
            line=dict(color="#4ade80", width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            height=350,
            yaxis_title="Unit Cost ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cost Variance Heatmap ──
    st.subheader("Cost Variance: Plant x Category")
    heatmap = load_cost_variance_heatmap()
    pivot = heatmap.pivot(index="plant", columns="category", values="variance_pct")
    fig = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Cost Variance % (Red = Over Budget)",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Commodity Prices ──
    st.subheader("Commodity Price Index (12 Months)")
    commodity = load_commodity_price_index()
    fig = px.line(
        commodity, x="month", y="price_index",
        color="commodity", title="Commodity Price Trends",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e1e2e",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
