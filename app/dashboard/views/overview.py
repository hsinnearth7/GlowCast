"""Executive Overview — cost analytics KPIs and business impact."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.data import (
    load_cost_overview,
    load_ocogs_trend,
    load_platform_flow,
    load_should_cost_breakdown,
    load_supplier_performance,
)


def render(metric_card) -> None:
    st.header("Executive Overview")
    kpi = load_cost_overview()

    # ── Top KPIs ──
    cols = st.columns(6)
    cards = [
        ("Cost Variance", f"{kpi['avg_cost_variance_pct']:.1%}", "Actual vs Budget"),
        ("Should-Cost Gap", f"{kpi['should_cost_gap_pct']:.1%}", "Opportunity to close"),
        ("Savings Realized", f"{kpi['savings_realized_pct']:.1%}", "YTD cost reduction"),
        ("Supplier OTD", f"{kpi['supplier_on_time_pct']:.1%}", "On-time delivery"),
        ("DoWhy ATE", f"${kpi['ate_cost_reduction']:.2f}", "Causal cost impact"),
        ("CUPED Reduction", f"{kpi['cuped_variance_reduction']:.0%}", "Experiment variance"),
    ]
    for col, (title, value, sub) in zip(cols, cards, strict=True):
        col.markdown(metric_card(title, value, sub), unsafe_allow_html=True)

    st.markdown("---")

    # ── Should-Cost Waterfall ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Should-Cost Breakdown")
        breakdown = load_should_cost_breakdown()
        fig = go.Figure(go.Waterfall(
            x=breakdown["element"],
            y=breakdown["gap"],
            textposition="outside",
            connector={"line": {"color": "#4a4a6a"}},
            increasing={"marker": {"color": "#f87171"}},
            decreasing={"marker": {"color": "#4ade80"}},
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            height=350,
            title="Cost Gap by Element ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Supplier Performance")
        suppliers = load_supplier_performance()
        fig = px.scatter(
            suppliers,
            x="quality_score", y="on_time_pct",
            size="total_orders", color="supplier",
            hover_data=["price_premium", "lead_time_days"],
            title="Quality vs Delivery (bubble = order volume)",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── OCOGS Trend ──
    st.subheader("OCOGS Trend (12 Months)")
    ocogs = load_ocogs_trend()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ocogs["month"], y=ocogs["avg_unit_cost"],
        mode="lines+markers", name="Actual OCOGS",
        line=dict(color="#60a5fa", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ocogs["month"], y=ocogs["target_cost"],
        mode="lines", name="Target Cost",
        line=dict(color="#4ade80", width=2, dash="dash"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e1e2e",
        height=300,
        yaxis_title="Unit Cost ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Data Flow Sankey ──
    st.subheader("Platform Data Flow")
    flow = load_platform_flow()
    labels = list(set(flow["source"].tolist() + flow["target"].tolist()))
    label_map = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color="#3d3d5c", pad=15, thickness=20),
        link=dict(
            source=[label_map[s] for s in flow["source"]],
            target=[label_map[t] for t in flow["target"]],
            value=flow["value"],
            color="rgba(96, 165, 250, 0.3)",
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        height=400,
        title="Cost Analytics Data Flow",
    )
    st.plotly_chart(fig, use_container_width=True)
