"""Inventory & Supply Chain — DoS/WoC, scrap risk, cross-zone, anomalies, social signals."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app.dashboard.data import (
    load_inventory_dos, load_scrap_risk, load_cross_zone,
    load_demand_anomalies, load_social_lead_lag, CONCERNS,
)


def render(metric_card):
    st.title("Inventory & Supply Chain")
    st.caption("FIFO optimization | Cross-zone penalties | Demand anomalies | Social signals")

    dos = load_inventory_dos()
    scrap = load_scrap_risk()
    cross = load_cross_zone()

    # ── KPI Row ──
    total_inv_value = dos["Inventory_Value"].sum()
    healthy_pct = (dos["Action"] == "Healthy").mean() * 100
    high_risk_pct = (dos["Action"] == "High_Risk").mean() * 100
    total_penalty = cross["Penalty_USD"].sum()
    avg_cross_pct = cross["Cross_Zone_Pct"].mean()
    critical_batches = (scrap["Risk_Tier"] == "Critical").sum()

    cols = st.columns(6)
    kpis = [
        ("Inventory Value", f"${total_inv_value/1e6:.1f}M", f"{len(dos)} records", "neutral", "#3498db"),
        ("Healthy Status", f"{healthy_pct:.0f}%", "DoS target", "good", "#2ecc71"),
        ("High Risk", f"{high_risk_pct:.0f}%", "Action needed", "bad", "#e74c3c"),
        ("Cross-Zone", f"{avg_cross_pct:.1f}%", f"Target <5%", "bad" if avg_cross_pct > 5 else "good", "#e67e22"),
        ("Penalties", f"${total_penalty/1e3:.0f}K", f"@$50/unit", "bad", "#9b59b6"),
        ("Critical Batches", f"{critical_batches}", "Expiry <7d", "bad" if critical_batches > 0 else "good", "#e74c3c"),
    ]
    for col, (label, value, delta, dtype, color) in zip(cols, kpis):
        col.markdown(metric_card(label, value, delta, dtype, color), unsafe_allow_html=True)

    st.markdown("")

    # ── Inventory DoS Heatmap ──
    st.markdown('<div class="section-header">Days of Supply Heatmap (FC x Concern)</div>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        action_filter = st.multiselect("Action Filter", ["Healthy", "Monitor", "High_Risk"],
                                        default=["Healthy", "Monitor", "High_Risk"])

    filtered_dos = dos[dos["Action"].isin(action_filter)]
    pivot_dos = filtered_dos.pivot_table(values="DoS", index="FC", columns="Concern", aggfunc="mean")

    fig_dos = go.Figure(data=go.Heatmap(
        z=pivot_dos.values, x=pivot_dos.columns.tolist(), y=pivot_dos.index.tolist(),
        colorscale="RdYlGn", text=pivot_dos.round(0).values,
        texttemplate="%{text:.0f}d", textfont=dict(size=11),
        zmin=10, zmax=70,
        colorbar=dict(title="DoS"),
    ))
    fig_dos.update_layout(
        height=420, margin=dict(l=80, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
    )
    st.plotly_chart(fig_dos, use_container_width=True)

    # ── Scrap Risk + Cross-Zone ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Scrap Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = scrap["Risk_Tier"].value_counts()
        risk_colors = {"Critical": "#e74c3c", "High": "#e67e22", "Medium": "#f39c12", "Low": "#2ecc71"}
        fig_risk = go.Figure(data=[go.Pie(
            labels=risk_counts.index, values=risk_counts.values,
            marker=dict(colors=[risk_colors.get(t, "#95a5a6") for t in risk_counts.index]),
            hole=0.45, textinfo="label+percent",
            textfont=dict(size=13),
        )])
        fig_risk.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1"),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Potential loss by concern
        loss_by_concern = scrap.groupby("Concern")["Potential_Loss"].sum().sort_values(ascending=False)
        fig_loss = go.Figure(go.Bar(
            x=loss_by_concern.index, y=loss_by_concern.values,
            marker_color=["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#9b59b6"],
            text=[f"${v/1e3:.1f}K" for v in loss_by_concern.values],
            textposition="outside",
        ))
        fig_loss.update_layout(
            yaxis_title="Potential Loss ($)", height=280,
            margin=dict(l=60, r=20, t=10, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"), yaxis=dict(gridcolor="#2d3348"),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Cross-Zone Penalties by FC</div>', unsafe_allow_html=True)
        cross_sorted = cross.sort_values("Penalty_USD", ascending=True)
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Bar(
            y=cross_sorted["FC"], x=cross_sorted["Penalty_USD"],
            orientation="h",
            marker_color=[
                "#e74c3c" if p > 200000 else "#e67e22" if p > 100000 else "#2ecc71"
                for p in cross_sorted["Penalty_USD"]
            ],
            text=[f"${v/1e3:.0f}K ({pct:.0f}%)" for v, pct in
                  zip(cross_sorted["Penalty_USD"], cross_sorted["Cross_Zone_Pct"])],
            textposition="outside",
        ))
        fig_cross.update_layout(
            xaxis_title="Penalty ($)", height=400,
            margin=dict(l=80, r=80, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"), xaxis=dict(gridcolor="#2d3348"),
        )
        st.plotly_chart(fig_cross, use_container_width=True)

        # Delivery time comparison
        fig_del = go.Figure()
        fig_del.add_trace(go.Bar(name="Local", y=cross_sorted["FC"],
                                  x=cross_sorted["Avg_Local_Days"],
                                  orientation="h", marker_color="#2ecc71"))
        fig_del.add_trace(go.Bar(name="Cross-Zone", y=cross_sorted["FC"],
                                  x=cross_sorted["Avg_Cross_Days"],
                                  orientation="h", marker_color="#e74c3c"))
        fig_del.update_layout(
            barmode="group", xaxis_title="Avg Delivery Days", height=300,
            margin=dict(l=80, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"), xaxis=dict(gridcolor="#2d3348"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_del, use_container_width=True)

    # ── Demand Anomalies ──
    st.markdown('<div class="section-header">Demand Anomaly Detection (Z-Score)</div>', unsafe_allow_html=True)
    anomalies = load_demand_anomalies()
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=anomalies["Date"], y=anomalies["Daily_Units"], mode="lines",
        line=dict(color="#3498db", width=1), name="Daily Demand",
    ))
    spikes = anomalies[anomalies["Anomaly"] == "Demand_Spike"]
    drops = anomalies[anomalies["Anomaly"] == "Demand_Drop"]
    fig_anom.add_trace(go.Scatter(
        x=spikes["Date"], y=spikes["Daily_Units"], mode="markers",
        marker=dict(color="#e74c3c", size=10, symbol="triangle-up"), name="Spike (Z>2)",
    ))
    fig_anom.add_trace(go.Scatter(
        x=drops["Date"], y=drops["Daily_Units"], mode="markers",
        marker=dict(color="#2ecc71", size=10, symbol="triangle-down"), name="Drop (Z<-2)",
    ))
    fig_anom.update_layout(
        yaxis_title="Units", height=300,
        margin=dict(l=60, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        xaxis=dict(gridcolor="#2d3348"), yaxis=dict(gridcolor="#2d3348"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # ── Social Lead-Lag ──
    st.markdown('<div class="section-header">Social Signal Cross-Correlation (T-k Lead)</div>', unsafe_allow_html=True)
    social = load_social_lead_lag()
    pivot_social = social.pivot_table(values="Pearson_R", index="Concern", columns="Lag_Days")

    fig_social = go.Figure(data=go.Heatmap(
        z=pivot_social.values, x=[f"T-{l}" for l in pivot_social.columns],
        y=pivot_social.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=pivot_social.round(2).values,
        texttemplate="%{text:.2f}", textfont=dict(size=10),
        zmin=-0.2, zmax=0.8,
        colorbar=dict(title="Pearson r"),
    ))
    fig_social.update_layout(
        height=280, margin=dict(l=120, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"), xaxis_title="Lag (days before sales)",
    )
    st.plotly_chart(fig_social, use_container_width=True)
