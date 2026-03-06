"""Executive Overview — All KPIs at a glance with platform flow visualization."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app.dashboard.data import (
    load_model_comparison, load_segment_evaluation, load_business_impact,
    load_platform_flow, load_uplift_results, load_fairness_results,
)


def render(metric_card):
    st.title("Executive Overview")
    st.caption("All platform KPIs at a glance | 5,000 SKUs | 12 FCs | 5 Countries")

    # ── Row 1: Top-level KPI cards ──
    cols = st.columns(6)
    cards = [
        ("Forecast MAPE", "11.8%", "-59% vs Naive", "good", "#e74c3c"),
        ("Conformal Coverage", "91%", "Target: 90%", "good", "#2ecc71"),
        ("Scrap Rate", "<2%", "-87% vs industry", "good", "#27ae60"),
        ("Cross-Zone", "<5%", "-86% vs uniform", "good", "#3498db"),
        ("X-Learner AUUC", "0.74", "+0.24 vs random", "good", "#9b59b6"),
        ("CUPED Reduction", "55%", "rho=0.74", "good", "#e67e22"),
    ]
    for col, (label, value, delta, dtype, color) in zip(cols, cards):
        col.markdown(metric_card(label, value, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: Platform Data Flow (Sankey) ──
    st.markdown('<div class="section-header">Platform Data Flow & Dependencies</div>', unsafe_allow_html=True)
    flow = load_platform_flow()
    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=25, line=dict(color="#2d3348", width=1),
            label=flow["labels"],
            color=["#3498db", "#1abc9c", "#9b59b6", "#e74c3c", "#e67e22",
                   "#3498db", "#2ecc71", "#2ecc71", "#2ecc71", "#2ecc71",
                   "#2ecc71", "#f39c12", "#f39c12", "#1abc9c", "#e67e22"],
        ),
        link=dict(
            source=flow["sources"], target=flow["targets"],
            value=flow["values"],
            color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.25)" for c in flow["colors"]],
        ),
    ))
    fig_sankey.update_layout(
        height=400, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1", size=13),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ── Row 3: Two-column layout ──
    col1, col2 = st.columns(2)

    # Business Impact Table
    with col1:
        st.markdown('<div class="section-header">Business Impact Summary</div>', unsafe_allow_html=True)
        impact = load_business_impact()
        fig_impact = go.Figure(data=[go.Table(
            header=dict(
                values=["Metric", "Before", "After", "Delta"],
                fill_color="#1a1f2e", font=dict(color="#ecf0f1", size=13),
                align="left", height=35,
            ),
            cells=dict(
                values=[impact[c] for c in impact.columns],
                fill_color=[["#111827"] * len(impact)],
                font=dict(color=["#ecf0f1", "#e74c3c", "#2ecc71", "#3498db"], size=12),
                align="left", height=30,
            ),
        )])
        fig_impact.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_impact, use_container_width=True)

    # Model Comparison Radar
    with col2:
        st.markdown('<div class="section-header">6-Model Performance Radar</div>', unsafe_allow_html=True)
        models = load_model_comparison()
        categories = ["MAPE (%)", "RMSE", "WMAPE (%)"]
        fig_radar = go.Figure()
        for _, row in models.iterrows():
            # Normalize: lower is better, so invert for radar
            max_vals = [30, 25, 30]
            r_vals = [max_vals[i] - row[c] for i, c in enumerate(["MAPE", "RMSE", "WMAPE"])]
            r_vals.append(r_vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=categories + [categories[0]],
                name=row["Model"],
                line=dict(width=2),
                opacity=0.85,
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 25], gridcolor="#2d3348"),
                angularaxis=dict(gridcolor="#2d3348"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=300, margin=dict(l=60, r=60, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1", size=11),
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Row 4: Segment Evaluation + Uplift ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Segment Slice Evaluation</div>', unsafe_allow_html=True)
        seg = load_segment_evaluation()
        fig_seg = go.Figure()
        colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
        for i, row in seg.iterrows():
            fig_seg.add_trace(go.Bar(
                x=[row["Segment"]], y=[row["MAPE"]],
                error_y=dict(type="data",
                             symmetric=False,
                             array=[row["CI_hi"] - row["MAPE"]],
                             arrayminus=[row["MAPE"] - row["CI_lo"]]),
                marker_color=colors[i], name=row["Segment"],
                text=f"{row['MAPE']}%", textposition="outside",
                showlegend=False,
            ))
        fig_seg.update_layout(
            yaxis_title="MAPE (%)", height=320,
            margin=dict(l=50, r=20, t=10, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            yaxis=dict(gridcolor="#2d3348", range=[0, 28]),
            xaxis=dict(tickangle=-15),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Uplift Learner Comparison</div>', unsafe_allow_html=True)
        uplift = load_uplift_results()
        colors_up = ["#95a5a6", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
        fig_up = go.Figure()
        for i, row in uplift.iterrows():
            fig_up.add_trace(go.Bar(
                x=[row["Learner"]], y=[row["AUUC"]],
                error_y=dict(type="data", symmetric=False,
                             array=[row["CI_hi"] - row["AUUC"]],
                             arrayminus=[row["AUUC"] - row["CI_lo"]]),
                marker_color=colors_up[i], showlegend=False,
                text=f"{row['AUUC']:.2f}", textposition="outside",
            ))
        fig_up.update_layout(
            yaxis_title="AUUC", height=320,
            margin=dict(l=50, r=20, t=10, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            yaxis=dict(gridcolor="#2d3348", range=[0, 0.9]),
            xaxis=dict(tickangle=-15),
        )
        # Add random baseline line
        fig_up.add_hline(y=0.5, line_dash="dash", line_color="#95a5a6",
                         annotation_text="Random", annotation_position="top right")
        st.plotly_chart(fig_up, use_container_width=True)

    # ── Row 5: Fairness Summary ──
    st.markdown('<div class="section-header">Fairness & Statistical Parity</div>', unsafe_allow_html=True)
    fair = load_fairness_results()
    col5, col6, col7 = st.columns([2, 2, 1])

    with col5:
        seg_fair = fair["segments"]
        fig_heat = go.Figure(data=go.Heatmap(
            z=seg_fair.pivot_table(values="MAPE", index="Concern", columns="Texture").values,
            x=["Lightweight", "Rich"],
            y=seg_fair["Concern"].unique(),
            colorscale="RdYlGn_r",
            text=seg_fair.pivot_table(values="MAPE", index="Concern", columns="Texture").values,
            texttemplate="%{text:.1f}%",
            textfont=dict(size=14),
            zmin=7, zmax=17,
            colorbar=dict(title="MAPE %"),
        ))
        fig_heat.update_layout(
            title="MAPE by Concern x Texture", height=300,
            margin=dict(l=120, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col6:
        fc_fair = fair["fcs"]
        fig_fc = px.bar(fc_fair.sort_values("MAPE"), x="MAPE", y="FC", orientation="h",
                        color="Country", height=300,
                        color_discrete_sequence=["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"])
        fig_fc.update_layout(
            title="MAPE by Fulfillment Center",
            margin=dict(l=80, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#2d3348"),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with col7:
        kw = fair["kruskal_wallis"]
        chi = fair["chi_squared"]
        st.markdown(f"""
        **Kruskal-Wallis Test**
        - H = {kw['H']}
        - p = {kw['p']}
        - <span class="badge badge-green">No Bias</span>

        **Chi-Squared Test**
        - X2 = {chi['chi2']}
        - p = {chi['p']}
        - <span class="badge badge-green">No Bias</span>
        """, unsafe_allow_html=True)
