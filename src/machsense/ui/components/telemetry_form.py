"""Interactive telemetry input form with live physical boundary validation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import streamlit as st

from machsense.data.schema import SensorBoundaries


def render_telemetry_form(
    initial_values: Optional[dict[str, Any]] = None,
) -> Tuple[Optional[dict[str, Any]], bool]:
    """Render interactive numeric sliders and inputs for machine telemetry.

    Args:
        initial_values: Optional preset values to pre-populate inputs.

    Returns:
        Tuple of (telemetry_dict, submit_button_clicked).
    """
    defaults = initial_values or {
        "type": "L",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }

    st.markdown("### 🎛️ Machine Sensor Telemetry Input")
    st.caption("Adjust real-time sensor parameters or load a physical failure preset from the sidebar.")

    with st.form("telemetry_input_form"):
        col1, col2 = st.columns(2)

        with col1:
            m_type = st.selectbox(
                "Product Quality Variant (Type)",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(defaults.get("type", "L")),
                help="L: Low Quality (60% volume), M: Medium Quality (30%), H: High Quality (10%)",
            )
            air_temp = st.slider(
                "Air Temperature [K]",
                min_value=float(SensorBoundaries.AIR_TEMP_MIN_K),
                max_value=float(SensorBoundaries.AIR_TEMP_MAX_K),
                value=float(defaults.get("air_temperature_k", 300.0)),
                step=0.1,
                help="Ambient environment temperature (270 K to 350 K)",
            )
            process_temp = st.slider(
                "Process Temperature [K]",
                min_value=float(SensorBoundaries.PROCESS_TEMP_MIN_K),
                max_value=float(SensorBoundaries.PROCESS_TEMP_MAX_K),
                value=float(defaults.get("process_temperature_k", 310.0)),
                step=0.1,
                help="Machining process chamber temperature (280 K to 360 K)",
            )

        with col2:
            speed = st.slider(
                "Spindle Rotational Speed [rpm]",
                min_value=float(SensorBoundaries.ROTATIONAL_SPEED_MIN_RPM),
                max_value=float(SensorBoundaries.ROTATIONAL_SPEED_MAX_RPM),
                value=float(defaults.get("rotational_speed_rpm", 1500.0)),
                step=10.0,
                help="Spindle rotational speed (500 to 4500 rpm)",
            )
            torque = st.slider(
                "Mechanical Torque [Nm]",
                min_value=float(SensorBoundaries.TORQUE_MIN_NM),
                max_value=float(SensorBoundaries.TORQUE_MAX_NM),
                value=float(defaults.get("torque_nm", 40.0)),
                step=0.5,
                help="Applied cutting tool resistance torque (0 to 150 Nm)",
            )
            tool_wear = st.slider(
                "Tool Wear Duration [min]",
                min_value=float(SensorBoundaries.TOOL_WEAR_MIN_MIN),
                max_value=float(SensorBoundaries.TOOL_WEAR_MAX_MIN),
                value=float(defaults.get("tool_wear_min", 50.0)),
                step=1.0,
                help="Cumulative active cutting tool wear time (0 to 500 min)",
            )

        # Live thermodynamic consistency check
        temp_delta = process_temp - air_temp
        if temp_delta < -0.5:
            st.warning(
                f"⚠️ **Thermodynamic Warning**: Process temperature ({process_temp:.1f} K) is lower than "
                f"air temperature ({air_temp:.1f} K). Active cutting generates heat ($T_{{process}} \\ge T_{{air}}$)."
            )
        elif temp_delta < 7.0:
            st.info(
                f"ℹ️ **Low Heat Dissipation**: Temperature delta is {temp_delta:.1f} K (cooling breakdown signature if speed is low)."
            )

        submitted = st.form_submit_button("🔍 Run Machine Health Diagnosis", type="primary", use_container_width=True)

    payload = {
        "type": m_type,
        "air_temperature_k": air_temp,
        "process_temperature_k": process_temp,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
    }

    return payload, submitted
