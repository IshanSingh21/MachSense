"""Batch CSV telemetry upload, high-throughput scoring, and reporting component for Streamlit."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from machsense.inference.schema import BatchPredictionResult, InferenceStatus
from machsense.ui.client import MachSenseUIClient


def render_batch_view(client: MachSenseUIClient, threshold: Optional[float] = None) -> None:
    """Render high-throughput fleet telemetry batch upload and analysis interface."""
    st.markdown("### 📁 High-Throughput Fleet Telemetry Processing")
    st.caption("Upload a CSV file containing machine sensor logs to perform vectorized batch failure screening.")

    uploaded_file = st.file_uploader(
        "Upload Telemetry CSV File",
        type=["csv"],
        help="CSV must contain: type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min",
    )

    if uploaded_file is None:
        st.info("👆 Upload a CSV file or download a template to test batch fleet inference.")
        # Provide sample template download
        sample_df = pd.DataFrame([
            {"type": "L", "air_temperature_k": 300.2, "process_temperature_k": 309.5, "rotational_speed_rpm": 1500.0, "torque_nm": 42.0, "tool_wear_min": 45.0},
            {"type": "M", "air_temperature_k": 302.5, "process_temperature_k": 311.8, "rotational_speed_rpm": 1380.0, "torque_nm": 58.5, "tool_wear_min": 210.0},
            {"type": "H", "air_temperature_k": 298.0, "process_temperature_k": 308.2, "rotational_speed_rpm": 2650.0, "torque_nm": 18.0, "tool_wear_min": 85.0},
        ])
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=sample_df.to_csv(index=False),
            file_name="machsense_batch_sample.csv",
            mime="text/csv",
        )
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse CSV file: {str(e)}")
        return

    st.markdown(f"**Loaded {len(raw_df)} records** from `{uploaded_file.name}`:")
    st.dataframe(raw_df.head(5), use_container_width=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("⚡ Score Fleet Telemetry", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Scoring {len(raw_df)} machine telemetry records..."):
            batch_result: BatchPredictionResult = client.predict_batch(
                df=raw_df,
                explain=False,
                threshold=threshold,
            )

        if batch_result.status == InferenceStatus.VALIDATION_ERROR:
            st.error("❌ **Batch Validation Failure**")
            for err in batch_result.errors:
                st.write(f"- {err}")
            return

        if batch_result.status == InferenceStatus.PROCESSING_ERROR:
            st.error("❌ **Batch Processing Failure**")
            for err in batch_result.errors:
                st.write(f"- {err}")
            return

        # 1. Summary Metrics
        st.success(f"✅ Successfully scored {batch_result.total_records} machine records in {batch_result.total_latency_ms:.1f} ms!")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Machines Scored", batch_result.total_records)
        with mcol2:
            st.metric(
                "Imminent Failures Flagged",
                batch_result.failure_count,
                delta=f"{(batch_result.failure_count / max(1, batch_result.total_records)) * 100.0:.1f}% flag rate",
                delta_color="inverse" if batch_result.failure_count > 0 else "normal",
            )
        with mcol3:
            st.metric(
                "Mean Fleet Failure Risk",
                f"{batch_result.mean_failure_probability * 100.0:.2f}%",
            )
        with mcol4:
            st.metric(
                "Fleet Processing Latency",
                f"{batch_result.total_latency_ms:.1f} ms",
                delta=f"{(batch_result.total_latency_ms / max(1, batch_result.total_records)):.2f} ms/rec",
                delta_color="off",
            )

        # 2. Enrich DataFrame with predictions
        enriched_df = raw_df.copy()
        enriched_df["failure_probability"] = [p.failure_probability for p in batch_result.predictions]
        enriched_df["predicted_class"] = [p.predicted_class for p in batch_result.predictions]
        enriched_df["predicted_label"] = [p.predicted_label for p in batch_result.predictions]
        enriched_df["risk_level"] = [p.risk_level.value if p.risk_level else "UNKNOWN" for p in batch_result.predictions]

        # 3. Interactive Data Table
        st.markdown("#### 📊 Enriched Machine Health Results")
        st.dataframe(
            enriched_df,
            use_container_width=True,
            column_config={
                "failure_probability": st.column_config.ProgressColumn(
                    "Failure Probability",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

        # 4. Download Enriched Results
        csv_data = enriched_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Enriched Fleet Predictions CSV",
            data=csv_data,
            file_name=f"machsense_predictions_{uploaded_file.name}",
            mime="text/csv",
            type="primary",
        )
