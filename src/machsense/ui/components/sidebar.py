"""Sidebar controls, model health indicators, and preset selectors for Streamlit."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import streamlit as st

from machsense import __version__
from machsense.ui.client import TELEMETRY_PRESETS, MachSenseUIClient


def render_sidebar(
    client: MachSenseUIClient,
) -> Tuple[Optional[dict[str, Any]], float, bool]:
    """Render sidebar controls.

    Returns:
        Tuple of (selected_preset_values_or_None, decision_threshold, explain_toggle).
    """
    with st.sidebar:
        st.title("⚙️ MachSense Controls")
        st.caption(f"Predictive Maintenance Suite v{__version__}")

        st.divider()

        # 1. System Health Status Indicator
        info = client.service_info
        st.markdown("#### 🟢 Model Status")
        st.markdown(
            f"""
            - **Model:** `{info.get('model_name', 'RandomForest')}`
            - **Version:** `{info.get('model_version', 'v1.0.0')}`
            - **SHAP Engine:** `{'Ready' if info.get('explainer_ready') else 'Disabled'}`
            """
        )

        st.divider()

        # 2. Decision Threshold Tuning Slider
        st.markdown("#### 🎚️ Operational Decision Threshold")
        default_thresh = float(client.optimal_threshold)
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.05,
            max_value=0.95,
            value=default_thresh,
            step=0.01,
            help=(
                f"Calibrated optimal F1 threshold is {default_thresh:.4f}. "
                "Lowering the threshold increases recall (catches more failures at the cost of potential false alarms)."
            ),
        )

        st.divider()

        # 3. Explainability Toggle
        explain_enabled = st.checkbox(
            "Enable TreeSHAP Explainability",
            value=True,
            help="Compute local feature attributions and operator diagnostics for each prediction.",
        )

        st.divider()

        # 4. Preset Scenario Quick-Load
        st.markdown("#### 📦 Quick-Load Failure Scenarios")
        st.caption("Populate inputs with real manufacturing failure patterns:")

        preset_names = list(TELEMETRY_PRESETS.keys())
        selected_preset_name = st.selectbox("Select Scenario Preset", ["-- Custom / Manual Entry --"] + preset_names)

        selected_preset = None
        if selected_preset_name != "-- Custom / Manual Entry --":
            selected_preset = TELEMETRY_PRESETS[selected_preset_name]
            st.info(f"ℹ️ **{selected_preset_name}**: {selected_preset.get('description', '')}")

        st.divider()

        # 5. Quick Links & Documentation
        st.markdown("#### 📚 References")
        st.markdown(
            """
            - [FastAPI Docs (Swagger)](http://127.0.0.1:8000/docs)
            - [Health Probe](http://127.0.0.1:8000/health)
            - [ReDoc Specs](http://127.0.0.1:8000/redoc)
            """
        )

    return selected_preset, threshold, explain_enabled
