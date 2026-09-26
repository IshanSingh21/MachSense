"""Reusable KPI and health status metric cards for Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from machsense.inference.schema import PredictionResult, RiskLevel


def render_prediction_metrics(result: PredictionResult, baseline_prob: float = 0.4988) -> None:
    """Render structured KPI cards displaying failure probability, risk level, and health state."""
    col1, col2, col3, col4 = st.columns(4)

    # 1. Health Status Banner
    with col1:
        if result.predicted_class == 0:
            st.metric(
                label="Machine Health State",
                value="HEALTHY",
                delta="Normal Operation",
                delta_color="normal",
            )
        else:
            st.metric(
                label="Machine Health State",
                value="FAILURE IMMINENT",
                delta="Immediate Action Required",
                delta_color="inverse",
            )

    # 2. Failure Probability with Baseline Comparison
    with col2:
        prob = result.failure_probability or 0.0
        delta_pct = (prob - baseline_prob) * 100.0
        st.metric(
            label="Failure Probability",
            value=f"{prob * 100.0:.1f}%",
            delta=f"{delta_pct:+.1f}% vs baseline",
            delta_color="inverse" if prob > baseline_prob else "normal",
        )

    # 3. Risk Classification Level
    with col3:
        risk_str = result.risk_level.value if result.risk_level else "UNKNOWN"
        st.metric(
            label="Risk Level",
            value=risk_str,
            delta=f"Threshold: {result.threshold_used:.3f}",
            delta_color="off",
        )

    # 4. Inference Latency & Model Version
    with col4:
        st.metric(
            label="Inference Latency",
            value=f"{result.latency_ms:.1f} ms",
            delta=f"Model: {result.model_version}",
            delta_color="off",
        )

    # Prominent visual status alert box
    if result.predicted_class == 1:
        if result.risk_level == RiskLevel.CRITICAL:
            st.error(
                "🚨 **CRITICAL RISK ALERT**: Machine telemetry indicates severe failure precursor signatures. "
                "Immediate inspection of mechanical and thermal systems is required."
            )
        else:
            st.warning(
                "⚠️ **ELEVATED RISK NOTICE**: Operating parameters exceed nominal safety envelopes. "
                "Schedule a preventive maintenance check during the next changeover."
            )
    else:
        if result.risk_level == RiskLevel.MODERATE_WARNING:
            st.info(
                "ℹ️ **MODERATE WARNING**: Machine telemetry is currently healthy, but minor parameter drift is observed."
            )
        else:
            st.success(
                "✅ **OPTIMAL OPERATING STATE**: Machine is operating safely within standard healthy boundaries."
            )
