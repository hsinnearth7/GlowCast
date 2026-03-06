"""Causal & Experimentation — DoWhy, uplift curves, CUPED, sequential testing, power analysis."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from app.dashboard.data import (
    load_dowhy_results, load_uplift_results, load_uplift_curve,
    load_cuped_results, load_sequential_test,
)


def render(metric_card):
    st.title("Causal & Experimentation")
    st.caption("DoWhy causal inference | Uplift modeling | CUPED | Sequential testing")

    # ── KPI Row ──
    dowhy = load_dowhy_results()
    cuped = load_cuped_results()
    uplift = load_uplift_results()

    cols = st.columns(6)
    kpis = [
        ("Causal ATE", f"{dowhy['ate']:.2f}", f"CI [{dowhy['ci_lower']:.2f}, {dowhy['ci_upper']:.2f}]", "good", "#3498db"),
        ("Refutations", "3/3 Passed", "Robust estimate", "good", "#2ecc71"),
        ("X-Learner", "0.74 AUUC", "Best learner", "good", "#e74c3c"),
        ("Treatment Split", "20/80", "Cost-efficient", "neutral", "#9b59b6"),
        ("CUPED rho", f"{cuped['rho']:.2f}", f"-{cuped['variance_reduction']*100:.0f}% variance", "good", "#e67e22"),
        ("N Observations", f"{dowhy['n_obs']/1000:.0f}K", "Sample size", "neutral", "#1abc9c"),
    ]
    for col, (label, value, delta, dtype, color) in zip(cols, kpis):
        col.markdown(metric_card(label, value, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("")

    # ── DoWhy + Uplift ──
    tab_causal, tab_experiment = st.tabs(["Causal Inference", "Experimentation"])

    with tab_causal:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">DoWhy 4-Step Pipeline</div>', unsafe_allow_html=True)

            # ATE visualization with CI
            fig_ate = go.Figure()
            fig_ate.add_trace(go.Bar(
                x=["ATE"], y=[dowhy["ate"]],
                error_y=dict(type="data", symmetric=False,
                             array=[dowhy["ci_upper"] - dowhy["ate"]],
                             arrayminus=[dowhy["ate"] - dowhy["ci_lower"]]),
                marker_color="#3498db", width=0.4,
                text=f"{dowhy['ate']:.2f}", textposition="outside",
            ))
            fig_ate.add_hline(y=0, line_dash="dash", line_color="#95a5a6")
            fig_ate.update_layout(
                yaxis_title="Average Treatment Effect (units)",
                height=250, margin=dict(l=60, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"), yaxis=dict(gridcolor="#2d3348"),
            )
            st.plotly_chart(fig_ate, use_container_width=True)

            # Refutation results
            st.markdown("**Refutation Tests**")
            for ref in dowhy["refutations"]:
                status = "badge-green" if ref["passed"] else "badge-red"
                icon = "Passed" if ref["passed"] else "Failed"
                st.markdown(
                    f'<span class="badge {status}">{icon}</span> '
                    f'**{ref["method"]}** | New ATE: {ref["new_ate"]:.2f} | p={ref["p_value"]:.2f}',
                    unsafe_allow_html=True,
                )

            # Pipeline steps visualization
            steps = ["1. Model (DAG)", "2. Identify (Backdoor)", "3. Estimate (OLS)", "4. Refute (3 tests)"]
            fig_steps = go.Figure(go.Funnel(
                y=steps, x=[100, 90, 85, 85],
                textinfo="text", text=steps,
                marker=dict(color=["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]),
            ))
            fig_steps.update_layout(
                height=200, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1"),
                showlegend=False,
            )
            st.plotly_chart(fig_steps, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Uplift Curves (4 Meta-Learners)</div>', unsafe_allow_html=True)

            curve_data = load_uplift_curve()
            learner_colors = {
                "Random": "#95a5a6", "S-Learner": "#3498db",
                "T-Learner": "#e67e22", "X-Learner": "#e74c3c",
                "Causal Forest": "#9b59b6",
            }
            fig_curve = go.Figure()
            for learner in ["Random", "S-Learner", "T-Learner", "Causal Forest", "X-Learner"]:
                dash = "dash" if learner == "Random" else "solid"
                width = 3 if learner == "X-Learner" else 2
                fig_curve.add_trace(go.Scatter(
                    x=curve_data["Fraction"], y=curve_data[learner],
                    mode="lines", name=learner,
                    line=dict(color=learner_colors[learner], width=width, dash=dash),
                ))
            fig_curve.update_layout(
                xaxis_title="Fraction Treated (cumulative)",
                yaxis_title="Normalized Uplift",
                height=350, margin=dict(l=60, r=20, t=10, b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
                xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
                legend=dict(x=0.05, y=0.95, bgcolor="rgba(0,0,0,0.3)"),
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # AUUC comparison
            ul = load_uplift_results()
            fig_auuc = go.Figure()
            auuc_colors = ["#95a5a6", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
            for i, row in ul.iterrows():
                fig_auuc.add_trace(go.Bar(
                    x=[row["Learner"]], y=[row["AUUC"]],
                    error_y=dict(type="data", symmetric=False,
                                 array=[row["CI_hi"] - row["AUUC"]],
                                 arrayminus=[row["AUUC"] - row["CI_lo"]]),
                    marker_color=auuc_colors[i], showlegend=False,
                    text=f"{row['AUUC']:.2f}", textposition="outside",
                ))
            fig_auuc.add_hline(y=0.5, line_dash="dash", line_color="#95a5a6",
                               annotation_text="Random baseline")
            fig_auuc.update_layout(
                yaxis_title="AUUC", height=280,
                margin=dict(l=60, r=20, t=10, b=80),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"), yaxis=dict(gridcolor="#2d3348", range=[0.3, 0.85]),
                xaxis=dict(tickangle=-15),
            )
            st.plotly_chart(fig_auuc, use_container_width=True)

    with tab_experiment:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-header">CUPED Variance Reduction</div>', unsafe_allow_html=True)

            # Variance reduction gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cuped["variance_reduction"] * 100,
                number=dict(suffix="%", font=dict(size=40)),
                delta=dict(reference=0, suffix="%"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#ecf0f1"),
                    bar=dict(color="#e67e22"),
                    bgcolor="#1a1f2e",
                    steps=[
                        dict(range=[0, 30], color="#2d3348"),
                        dict(range=[30, 50], color="#34495e"),
                        dict(range=[50, 100], color="#1a1f2e"),
                    ],
                    threshold=dict(line=dict(color="#2ecc71", width=3), thickness=0.8, value=55),
                ),
                title=dict(text="Variance Reduction", font=dict(size=16)),
            ))
            fig_gauge.update_layout(
                height=250, margin=dict(l=30, r=30, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Sample size table
            st.markdown("**Sample Size Impact**")
            cuped_table = cuped["table"]
            fig_table = go.Figure()
            fig_table.add_trace(go.Bar(
                name="Raw", x=cuped_table["MDE"], y=cuped_table["n_raw"],
                marker_color="#e74c3c", text=[f"{v:,}" for v in cuped_table["n_raw"]],
                textposition="outside",
            ))
            fig_table.add_trace(go.Bar(
                name="CUPED", x=cuped_table["MDE"], y=cuped_table["n_cuped"],
                marker_color="#2ecc71", text=[f"{v:,}" for v in cuped_table["n_cuped"]],
                textposition="outside",
            ))
            fig_table.update_layout(
                barmode="group", xaxis_title="Minimum Detectable Effect",
                yaxis_title="Sample Size (per group)", height=300,
                margin=dict(l=60, r=20, t=10, b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"), yaxis=dict(gridcolor="#2d3348"),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_table, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">Sequential Testing (mSPRT)</div>', unsafe_allow_html=True)

            seq = load_sequential_test()
            fig_seq = go.Figure()

            # P-value trajectory
            fig_seq.add_trace(go.Scatter(
                x=seq["N_total"], y=seq["P_value"], mode="lines",
                name="p-value", line=dict(color="#3498db", width=2),
            ))
            fig_seq.add_hline(y=0.05, line_dash="dash", line_color="#e74c3c",
                              annotation_text="alpha=0.05")

            # Shade stopping region
            stop_idx = seq[seq["Stopped"]].index
            if len(stop_idx) > 0:
                stop_n = seq.loc[stop_idx[0], "N_total"]
                fig_seq.add_vline(x=stop_n, line_dash="dot", line_color="#2ecc71",
                                  annotation_text=f"Stop @ N={stop_n:,}")

            fig_seq.update_layout(
                xaxis_title="Total Sample Size", yaxis_title="p-value",
                yaxis_type="log", height=300,
                margin=dict(l=60, r=20, t=10, b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
                xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_seq, use_container_width=True)

            # Observed delta trajectory
            st.markdown("**Observed Treatment Effect Over Time**")
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=seq["N_total"], y=seq["Observed_Delta"], mode="lines",
                line=dict(color="#e67e22", width=2), name="Observed Delta",
            ))
            fig_delta.add_hline(y=0.03, line_dash="dash", line_color="#2ecc71",
                                annotation_text="True effect (3%)")
            fig_delta.add_hline(y=0, line_dash="dot", line_color="#95a5a6")
            fig_delta.update_layout(
                xaxis_title="Total Sample Size", yaxis_title="Observed Delta",
                height=250, margin=dict(l=60, r=20, t=10, b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
                xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
            )
            st.plotly_chart(fig_delta, use_container_width=True)

    # ── Methodology Summary ──
    st.markdown('<div class="section-header">Methodology Interconnections</div>', unsafe_allow_html=True)
    fig_method = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="#2d3348", width=1),
            label=["Observational Data", "Treatment Assignment (SHA-256)",
                   "DoWhy (ATE)", "Uplift (CATE)", "CUPED", "Sequential Test",
                   "Promotion Strategy", "Budget Allocation"],
            color=["#3498db", "#e67e22", "#9b59b6", "#e74c3c",
                   "#2ecc71", "#1abc9c", "#f39c12", "#27ae60"],
        ),
        link=dict(
            source=[0, 1, 1, 0, 0, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5, 6, 7, 6, 6],
            value=[30, 15, 15, 10, 10, 12, 12, 8, 8],
            color=["rgba(52,152,219,0.3)", "rgba(230,126,34,0.3)", "rgba(231,76,60,0.3)",
                   "rgba(46,204,113,0.3)", "rgba(26,188,156,0.3)", "rgba(155,89,182,0.3)",
                   "rgba(231,76,60,0.3)", "rgba(46,204,113,0.3)", "rgba(26,188,156,0.3)"],
        ),
    ))
    fig_method.update_layout(
        height=300, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1", size=12),
    )
    st.plotly_chart(fig_method, use_container_width=True)
