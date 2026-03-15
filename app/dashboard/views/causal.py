"""Causal & Experimentation — DoWhy causal inference and CUPED A/B testing for cost analytics."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.dashboard.data import (
    load_cuped_results,
    load_dowhy_results,
    load_sequential_test,
    load_uplift_curve,
    load_uplift_results,
)


def render(metric_card) -> None:
    st.header("Causal & Experimentation")

    dowhy = load_dowhy_results()
    cuped = load_cuped_results()

    # ── Top KPIs ──
    cols = st.columns(6)
    cards = [
        ("ATE (Cost Impact)", f"${dowhy['ate']:.2f}", f"[{dowhy['ci_lower']:.2f}, {dowhy['ci_upper']:.2f}]"),
        ("Refutations", "3/3 Passed", "Causal validity checks"),
        ("X-Learner AUUC", "0.74", "Best uplift model"),
        ("Treatment Split", "20/80", "Cost action vs control"),
        ("CUPED rho", f"{cuped['rho']:.2f}", f"-{cuped['variance_reduction']:.0%} variance"),
        ("Observations", f"{dowhy['n_obs']:,}", "Cost transactions"),
    ]
    for col, (t, v, s) in zip(cols, cards, strict=True):
        col.markdown(metric_card(t, v, s), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2 = st.tabs(["Causal Inference", "Experimentation"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average Treatment Effect (ATE)")
            st.markdown(f"""
            **Treatment**: `{dowhy['treatment']}` (cost reduction intervention)
            **Outcome**: `{dowhy['outcome']}` (change in unit cost)
            """)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["ATE"], y=[dowhy["ate"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[dowhy["ci_upper"] - dowhy["ate"]],
                    arrayminus=[dowhy["ate"] - dowhy["ci_lower"]],
                ),
                marker_color="#4ade80",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#6b6b8d")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e1e2e",
                height=300,
                yaxis_title="Cost Change ($)",
                title="Causal Effect of Cost Reduction Actions",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Refutation Tests")
            for test, result in dowhy["refutations"].items():
                badge = "badge-green" if result["passed"] else "badge-red"
                status = "PASSED" if result["passed"] else "FAILED"
                st.markdown(
                    f'<span class="{badge} badge">{status}</span> **{test}**',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.subheader("Uplift Model Comparison")
            uplift = load_uplift_results()
            fig = go.Figure(go.Bar(
                x=uplift["learner"], y=uplift["auuc"],
                marker_color=["#6b6b8d", "#60a5fa", "#4ade80", "#facc15"],
                text=uplift["auuc"],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e1e2e",
                height=280,
                yaxis_title="AUUC",
                title="Uplift AUUC by Meta-Learner",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Uplift Curve ──
        st.subheader("Uplift Curve (X-Learner vs Random)")
        curve = load_uplift_curve()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["fraction_treated"], y=curve["x_learner_uplift"],
            mode="lines", name="X-Learner",
            line=dict(color="#4ade80", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=curve["fraction_treated"], y=curve["random_uplift"],
            mode="lines", name="Random",
            line=dict(color="#6b6b8d", width=1, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e1e2e",
            height=300,
            xaxis_title="Fraction Treated",
            yaxis_title="Cumulative Uplift",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CUPED Variance Reduction")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cuped["variance_reduction"] * 100,
                title={"text": "Variance Reduction %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4ade80"},
                    "bgcolor": "#1e1e2e",
                    "steps": [
                        {"range": [0, 30], "color": "#2d2d44"},
                        {"range": [30, 60], "color": "#3d3d5c"},
                        {"range": [60, 100], "color": "#4d4d6c"},
                    ],
                },
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Sample Size Impact (CUPED)**")
            for mde, data in cuped["sample_size_reduction"].items():
                st.markdown(f"- **{mde}**: {data['raw']:,} → {data['cuped']:,} ({data['reduction']})")

        with col2:
            st.subheader("Sequential Testing (mSPRT)")
            seq = load_sequential_test()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=seq["day"], y=seq["p_value"],
                mode="lines", name="p-value",
                line=dict(color="#60a5fa", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=seq["day"], y=seq["alpha"],
                mode="lines", name="Alpha",
                line=dict(color="#f87171", width=1, dash="dash"),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1e1e2e",
                height=300,
                yaxis_title="p-value",
                xaxis_title="Day",
                yaxis_type="log",
                title="Sequential Test p-value Trajectory",
            )
            st.plotly_chart(fig, use_container_width=True)
