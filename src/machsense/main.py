"""Main entry point and CLI for MachSense."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from machsense import __version__
from machsense.config.settings import get_project_root, get_settings, load_config
from machsense.utils.logger import get_logger, setup_logging


def print_banner() -> None:
    """Display project header banner."""
    banner = r"""
===================================================================
   __  __            _     ____
  |  \/  | __ _  ___| |__ / ___|  ___ _ __  ___  ___
  | |\/| |/ _` |/ __| '_ \\___ \ / _ \ '_ \/ __|/ _ \
  | |  | | (_| | (__| | | |___) |  __/ | | \__ \  __/
  |_|  |_|\__,_|\___|_| |_|____/ \___|_| |_|___/\___|
  AI-Powered Predictive Maintenance System (v""" + f"""{__version__})
===================================================================
"""
    print(banner)


def show_info(config_path: str | None = None) -> int:
    """Print project metadata, resolved paths, and configuration summary."""
    setup_logging()
    logger = get_logger("machsense.cli")

    print_banner()
    logger.info("Initializing MachSense environment check...")

    root = get_project_root()
    settings = load_config(config_path) if config_path else get_settings()

    print(f"Project Name:        {settings.project.name}")
    print(f"Version:             {settings.project.version}")
    print(f"Environment:         {settings.project.environment}")
    print(f"Project Root:        {root}")
    print(f"Raw Data Path:       {settings.resolve_path('raw_data_dir')}")
    print(f"Processed Data Path: {settings.resolve_path('processed_data_dir')}")
    print(f"Models Path:         {settings.resolve_path('models_dir')}")
    print(f"Logs Path:           {settings.resolve_path('logs_dir')}")
    print(f"Configs Path:        {settings.resolve_path('configs_dir')}")
    print("-------------------------------------------------------------------")
    print(f"Target Column:       {settings.data.target_column}")
    print(f"Features ({len(settings.features.numerical_features)} numerical): {', '.join(settings.features.numerical_features)}")
    print(f"Baseline Model:      {settings.model.name} ({settings.model.type})")
    print(f"Explainability:      {settings.model.explainability.method.upper()} (Enabled: {settings.model.explainability.enabled})")
    print("===================================================================")

    logger.info("MachSense foundation check completed successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="machsense",
        description="MachSense: AI-Powered Predictive Maintenance System",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MachSense {__version__}",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display resolved project configuration and environment status.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run SHAP explainability analysis on champion model and test telemetry.",
    )
    parser.add_argument(
        "--predict-sample",
        action="store_true",
        help="Run end-to-end production inference pipeline on a sample sensor payload.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the MachSense FastAPI backend server with Uvicorn.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override server bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port (default: 8000).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the MachSense interactive Streamlit user dashboard.",
    )
    return parser


def run_ui_cli() -> int:
    """Launch the Streamlit user dashboard."""
    import subprocess
    from machsense.config.settings import get_project_root

    root = get_project_root()
    app_path = root / "src" / "machsense" / "ui" / "app.py"

    print_banner()
    logger = get_logger("machsense.cli")
    logger.info("Launching MachSense Streamlit User Dashboard...")
    print(f"Starting Streamlit from: {app_path}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


def run_explainability_cli(config_path: str | None = None) -> int:
    """Run SHAP explainability CLI demonstration."""
    setup_logging()
    logger = get_logger("machsense.cli")
    logger.info("Executing MachSense SHAP Explainability CLI...")
    from machsense.models.explainability import main as explain_main
    explain_main()
    return 0


def run_server_cli(
    config_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> int:
    """Launch the FastAPI server using Uvicorn."""
    import uvicorn

    setup_logging()
    logger = get_logger("machsense.cli")
    settings = load_config(config_path) if config_path else get_settings()

    bind_host = host or settings.serving.host
    bind_port = port or settings.serving.port

    print_banner()
    logger.info("Starting MachSense FastAPI backend on http://%s:%d ...", bind_host, bind_port)
    print(f"API Documentation available at: http://{bind_host}:{bind_port}/docs")
    print(f"ReDoc Documentation at:         http://{bind_host}:{bind_port}/redoc")
    print(f"Health Check at:                http://{bind_host}:{bind_port}/health")

    uvicorn.run(
        "machsense.api.app:app",
        host=bind_host,
        port=bind_port,
        reload=settings.serving.reload,
        workers=settings.serving.workers if not settings.serving.reload else 1,
    )
    return 0


def run_predict_sample_cli(config_path: str | None = None) -> int:
    """Run live sample inference through MachSensePredictionService."""
    setup_logging()
    logger = get_logger("machsense.cli")
    logger.info("Executing MachSense Prediction Pipeline demonstration...")

    from machsense.inference.pipeline import MachSensePredictionService

    service = MachSensePredictionService()
    sample_payload = {
        "type": "L",
        "air_temperature_k": 302.5,
        "process_temperature_k": 311.8,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 58.5,
        "tool_wear_min": 210.0,
    }

    print("\n--- Incoming Sensor Payload ---")
    for k, v in sample_payload.items():
        print(f"  {k}: {v}")

    result = service.predict(sample_payload, explain=True)

    print("\n--- MachSense Production Prediction Result ---")
    print(f"Status:              {result.status.value}")
    print(f"Failure Probability: {result.failure_probability:.4%}")
    print(f"Predicted Class:     {result.predicted_class} ({result.predicted_label})")
    print(f"Risk Level:          {result.risk_level.value}")
    print(f"Threshold Applied:   {result.threshold_used:.4f}")
    print(f"Model Version:       {result.model_version}")
    print(f"Latency:             {result.latency_ms:.2f} ms")
    print("\n--- Operator Diagnostic Summary ---")
    print(result.operator_summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Application CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ui:
        return run_ui_cli()

    if args.serve:
        return run_server_cli(config_path=args.config, host=args.host, port=args.port)

    if args.predict_sample:
        return run_predict_sample_cli(config_path=args.config)

    if args.explain:
        return run_explainability_cli(config_path=args.config)

    if args.info or len(sys.argv) == 1 or (argv is not None and len(argv) == 0):
        return show_info(config_path=args.config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
