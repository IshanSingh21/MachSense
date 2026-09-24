"""MachSense Streamlit Dashboard: Industrial Predictive Maintenance & Diagnostics Suite."""

from __future__ import annotations

import streamlit as st

from machsense import __version__
from machsense.inference.schema import InferenceStatus, PredictionResult
from machsense.ui.client import MachSenseUIClient
from machsense.ui.components import (
    render_batch_view,
    render_operator_diagnostics,
    render_prediction_metrics,
    render_shap_attribution_chart,
    render_sidebar,
    render_telemetry_form,
)

# 1. Page Configuration
st.set_page_config(
    page_title="MachSense | Predictive Maintenance Suite",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom Modern Industrial CSS Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MachSense AI model & SHAP explainer...")
def get_ui_client() -> MachSenseUIClient:
    """Initialize singleton UI client with cached memory-loaded artifacts."""
    return MachSenseUIClient()


def main() -> None:
    """Main dashboard entry point."""
    client = get_ui_client()

    # Sidebar Controls
    selected_preset, threshold, explain_enabled = render_sidebar(client)

    # App Header
    st.markdown('<div class="main-header">⚙️ MachSense: Predictive Maintenance Suite</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">AI-Powered Machine Health Monitoring, Early Failure Prediction & TreeSHAP Diagnostics (v{__version__})</div>',
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs
    tab_single, tab_batch, tab_guide = st.tabs([
        "🔬 Real-Time Machine Diagnostics",
        "📁 Fleet Batch Telemetry Processing",
        "📖 Physical Failure Regimes & Model Card",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Real-Time Machine Diagnostics
    # -----------------------------------------------------------------------
    with tab_single:
        col_input, col_output = st.columns([1, 1.2], gap="medium")

        with col_input:
            payload, submitted = render_telemetry_form(initial_values=selected_preset)

        with col_output:
            st.markdown("### 📊 Health Diagnostic & Explainability")

            # Execute prediction on form submit or on first load with preset
            if submitted or "last_result" in st.session_state:
                with st.spinner("Analyzing telemetry & computing Shapley attributions..."):
                    result: PredictionResult = client.predict_single(
                        payload=payload,
                        explain=explain_enabled,
                        threshold=threshold,
                    )
                    st.session_state["last_result"] = result

                if result.status == InferenceStatus.VALIDATION_ERROR:
                    st.error("❌ **Telemetry Validation Failure**")
                    for err in result.errors:
                        st.write(f"- {err}")
                elif result.status == InferenceStatus.PROCESSING_ERROR:
                    st.error("❌ **Inference Processing Error**")
                    for err in result.errors:
                        st.write(f"- {err}")
                else:
                    # 1. KPI Metric Cards
                    render_prediction_metrics(result)

                    st.divider()

                    # 2. SHAP Feature Attribution Chart
                    if explain_enabled and (result.top_risk_escalators or result.top_stabilizers):
                        render_shap_attribution_chart(result)

                    # 3. Plain-English Operator Diagnostics & Safety Notice
                    render_operator_diagnostics(result)
            else:
                st.info("👈 Set sensor parameters and click **'Run Machine Health Diagnosis'** to evaluate machine health.")

    # -----------------------------------------------------------------------
    # TAB 2: Fleet Batch Telemetry Processing
    # -----------------------------------------------------------------------
    with tab_batch:
        render_batch_view(client=client, threshold=threshold)

    # -----------------------------------------------------------------------
    # TAB 3: Physical Failure Regimes & Model Card
    # -----------------------------------------------------------------------
    with tab_guide:
        st.markdown("### 🔬 Domain Physical Failure Regimes")
        st.markdown(
            """
            MachSense monitors CNC milling and machining telemetry, decomposing sensor signals across five physical degradation modes:
            """
        )

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown(
                """
                #### 1. Heat Dissipation Failure (HDF)
                - **Physical Cause**: Cooling system breakdown or lubricant starvation.
                - **Signature**: Air-Process temperature difference $\Delta T < 8.6\\text{ K}$ combined with spindle speeds below $1380\\text{ rpm}$.
                - **Detection Rate**: **100%**.

                #### 2. Power Failure (PWF)
                - **Physical Cause**: Motor drive electrical overload or mechanical resistance stall.
                - **Signature**: Power wattage $P = \\tau \\cdot \\omega$ outside $3500\\text{ W} - 9000\\text{ W}$.
                - **Detection Rate**: **100%**.

                #### 3. Overstrain Failure (OSF)
                - **Physical Cause**: Excessive cutting torque applied to a structurally degraded, worn tool.
                - **Signature**: Overstrain Index $(\\text{Torque} \\times \\text{Tool Wear}) > 11,000\\text{ Nm}\\cdot\\text{min}$ (Type L) or $> 12,000$ (Type M).
                - **Detection Rate**: **100%**.
                """
            )
        with f_col2:
            st.markdown(
                """
                #### 4. Tool Wear Failure (TWF)
                - **Physical Cause**: Cutting insert flank wear exceeding permissible clearance.
                - **Signature**: Cumulative tool wear time $t_{\\text{wear}} > 200–240\\text{ min}$.
                - **Detection Rate**: **85.7%** (Calibrating threshold to $0.35$ raises detection to $>94\%$).

                #### 5. Random Failures (RNF)
                - **Physical Cause**: True stochastic hardware fault (e.g. electrical surge).
                - **Signature**: No pre-failure sensor telemetry anomalies.
                - **Detection Rate**: **0%** (Governed by sensor data physical limits).
                """
            )

        st.divider()
        st.markdown("### 🏅 Active Champion Model Card (`v1.0.0`)")
        meta = client.service_info
        st.json(meta)


if __name__ == "__main__":
    main()
