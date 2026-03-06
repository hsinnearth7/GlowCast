"""Demand & Forecasting — Model comparison, walk-forward CV, ablation study."""

import streamlit as st
import plotly.graph_objects as go
from app.dashboard.data import (
    load_model_comparison, load_walkforward_cv,
    load_ablation_study, MODEL_COLORS, SEGMENT_GENES,
)


def render(metric_card):
    st.title("Demand & Forecasting")
    st.caption("6-model routing ensemble | Walk-forward CV | Ablation study")

    # ── KPI Row ──
    cols = st.columns(6)
    kpis = [
        ("Best Single", "12.5%", "LightGBM", "neutral", "#9b59b6"),
        ("Ensemble", "11.8%", "Routing", "good", "#e74c3c"),
        ("vs Naive", "-59%", "Improvement", "good", "#2ecc71"),
        ("Coverage", "91%", "90% target", "good", "#3498db"),
        ("CV Folds", "12", "Monthly retrain", "neutral", "#e67e22"),
        ("Horizon", "14d", "Test window", "neutral", "#1abc9c"),
    ]
    for col, (label, value, delta, dtype, color) in zip(cols, kpis):
        col.markdown(metric_card(label, value, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("")

    # ── Model Comparison ──
    st.markdown('<div class="section-header">6-Model Performance Comparison</div>', unsafe_allow_html=True)

    models = load_model_comparison()
    tab_bar, tab_grouped, tab_table = st.tabs(["MAPE Ranking", "All Metrics", "Data Table"])

    with tab_bar:
        sorted_models = models.sort_values("MAPE")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_models["Model"], y=sorted_models["MAPE"],
            marker_color=[MODEL_COLORS.get(m, "#95a5a6") for m in sorted_models["Model"]],
            text=[f"{v:.1f}%" for v in sorted_models["MAPE"]],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="MAPE (%)", height=380,
            margin=dict(l=50, r=20, t=20, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            yaxis=dict(gridcolor="#2d3348", range=[0, 35]),
        )
        # Target line
        fig.add_hline(y=12, line_dash="dash", line_color="#2ecc71",
                      annotation_text="Target 12%", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

    with tab_grouped:
        fig_g = go.Figure()
        for metric, color in [("MAPE", "#e74c3c"), ("RMSE", "#3498db"), ("WMAPE", "#2ecc71")]:
            fig_g.add_trace(go.Bar(
                name=metric, x=models["Model"], y=models[metric],
                marker_color=color, opacity=0.85,
                text=[f"{v:.1f}" for v in models[metric]], textposition="outside",
            ))
        fig_g.update_layout(
            barmode="group", height=380, yaxis_title="Value",
            margin=dict(l=50, r=20, t=20, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            yaxis=dict(gridcolor="#2d3348"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with tab_table:
        st.dataframe(models.style.format({"MAPE": "{:.1f}%", "RMSE": "{:.1f}", "WMAPE": "{:.1f}%"}),
                      use_container_width=True)

    # ── Walk-Forward CV ──
    st.markdown('<div class="section-header">12-Fold Walk-Forward Cross-Validation</div>', unsafe_allow_html=True)

    cv = load_walkforward_cv()
    selected_models = st.multiselect(
        "Select models", _MODELS := list(MODEL_COLORS.keys()),
        default=["Routing Ensemble", "LightGBM", "NaiveMA(30)"],
    )

    filtered = cv[cv["Model"].isin(selected_models)]
    fig_cv = go.Figure()
    for model in selected_models:
        mdf = filtered[filtered["Model"] == model]
        fig_cv.add_trace(go.Scatter(
            x=mdf["Fold"], y=mdf["MAPE"], mode="lines+markers",
            name=model, line=dict(color=MODEL_COLORS.get(model, "#95a5a6"), width=2),
            marker=dict(size=6),
        ))

    fig_cv.update_layout(
        xaxis_title="Fold (0=most recent)", yaxis_title="MAPE (%)", height=350,
        margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # ── Routing Ensemble Logic + Ablation ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Routing Ensemble Decision Logic</div>', unsafe_allow_html=True)
        # Treemap showing routing logic
        routing_data = [
            {"Route": "Mature SKUs", "Model": "LightGBM", "Pct": 65, "Condition": "Default"},
            {"Route": "Cold Start (<60d)", "Model": "Chronos-2 ZS", "Pct": 12, "Condition": "history < 60 days"},
            {"Route": "Intermittent (CV>1.5)", "Model": "SARIMAX", "Pct": 23, "Condition": "CV(demand) > 1.5"},
        ]
        import pandas as pd
        rdf = pd.DataFrame(routing_data)
        fig_route = go.Figure(go.Treemap(
            labels=[f"{r['Route']}<br>{r['Model']}<br>{r['Pct']}%" for _, r in rdf.iterrows()],
            parents=[""] * 3,
            values=rdf["Pct"],
            marker=dict(colors=["#9b59b6", "#1abc9c", "#3498db"]),
            textinfo="label",
            textfont=dict(size=14),
        ))
        fig_route.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_route, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Feature Ablation Study (LightGBM)</div>', unsafe_allow_html=True)
        ablation = load_ablation_study()
        colors_abl = []
        for _, row in ablation.iterrows():
            sig = row["Significance"]
            if sig == "Baseline":
                colors_abl.append("#3498db")
            elif "***" in str(sig):
                colors_abl.append("#e74c3c")
            elif "**" in str(sig):
                colors_abl.append("#e67e22")
            elif "*" in str(sig) and "n.s." not in str(sig):
                colors_abl.append("#f39c12")
            else:
                colors_abl.append("#95a5a6")

        fig_abl = go.Figure(go.Bar(
            x=ablation["MAPE"], y=ablation["Configuration"],
            orientation="h", marker_color=colors_abl,
            text=[f"{v:.1f}% (+{d:.1f})" if d > 0 else f"{v:.1f}%"
                  for v, d in zip(ablation["MAPE"], ablation["Delta"])],
            textposition="outside",
        ))
        fig_abl.add_vline(x=12.5, line_dash="dash", line_color="#3498db",
                          annotation_text="Full Model", annotation_position="top right")
        fig_abl.update_layout(
            xaxis_title="MAPE (%)", height=320,
            margin=dict(l=180, r=60, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#2d3348", range=[10, 18]),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

    # ── Segment Heatmap ──
    st.markdown('<div class="section-header">MAPE by Segment Gene (Concern x Texture)</div>', unsafe_allow_html=True)
    import pandas as pd
    seg_data = []
    for (concern, texture), gene in SEGMENT_GENES.items():
        seg_data.append({"Concern": concern, "Texture": texture, "MAPE_Target": gene["mape_target"]})
    sdf = pd.DataFrame(seg_data)
    pivot = sdf.pivot_table(values="MAPE_Target", index="Concern", columns="Texture")

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn_r", text=pivot.values,
        texttemplate="%{text:.1f}%", textfont=dict(size=15),
        zmin=7, zmax=17,
        colorbar=dict(title="MAPE %"),
    ))
    fig_heat.update_layout(
        height=300, margin=dict(l=120, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
