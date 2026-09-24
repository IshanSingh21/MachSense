"""Interactive SHAP attribution charts and operator diagnostic components for Streamlit."""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from machsense.inference.schema import PredictionResult


def render_shap_attribution_chart(result: PredictionResult) -> None:
    """Render interactive Plotly horizontal bar chart of SHAP risk escalators and stabilizers."""
    escalators = result.top_risk_escalators or []
    stabilizers = result.top_stabilizers or []

    if not escalators and not stabilizers:
        st.info("No feature attributions available for this prediction.")
        return

    # Combine and sort for chart
    all_features = []
    for esc in escalators:
        all_features.append({
            "name": esc.get("display_name", esc.get("feature")),
            "shap": float(esc.get("shap_value", 0.0)),
            "type": "Risk Escalator (Pushing toward failure)",
            "color": "#EF4444",  # Crimson / Red
        })
    for stb in stabilizers:
        all_features.append({
            "name": stb.get("display_name", stb.get("feature")),
            "shap": float(stb.get("shap_value", 0.0)),
            "type": "Stabilizer (Maintaining safety margin)",
            "color": "#10B981",  # Emerald / Green
        })

    df = pd.DataFrame(all_features).sort_values(by="shap", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["name"],
            x=df["shap"],
            orientation="h",
            marker=dict(
                color=df["color"],
                line=dict(color="#1E293B", width=1),
            ),
            hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Local Feature Attributions (TreeSHAP)</b>",
        xaxis_title="SHAP Value (Impact on Failure Probability Score)",
        yaxis_title="",
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=40, b=30),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#64748B"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_operator_diagnostics(result: PredictionResult) -> None:
    """Render plain-English operator summary and safety disclaimers."""
    st.markdown("### 📋 Shop-Floor Operator Diagnostics")

    if result.operator_summary:
        st.text_area(
            label="Automated Diagnostic Summary",
            value=result.operator_summary,
            height=200,
            disabled=True,
            help="Plain-English maintenance instructions generated directly from TreeSHAP decision paths.",
        )

    with st.expander("🛡️ Non-Causal Safety Notice & Physical Inspection Guidance", expanded=False):
        st.caption(result.non_causal_disclaimer)
        st.markdown(
            """
            **Recommended Operator Protocols:**
            1. **Verify Physical Systems**: Always perform visual and mechanical checks (e.g. check lubrication lines, tool insert wear, and spindle bearings) prior to executing intrusive maintenance.
            2. **Correlated Subsystems**: Electrical wattage, torque, and rotational speed are coupled variables. Inspect the motor drive train as an integrated system.
            3. **Sensor Faults**: If values deviate wildly from historical norms without physical symptoms, inspect sensor cabling and RTD calibration.
            """
        )
