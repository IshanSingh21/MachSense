"""Unit tests for MachSense CLI entry point."""

import pytest
from machsense.main import build_parser, main


def test_cli_parser_defaults():
    """Verify CLI parser argument defaults."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.info is False
    assert args.config is None


def test_cli_parser_info_flag():
    """Verify CLI parser accepts --info flag."""
    parser = build_parser()
    args = parser.parse_args(["--info"])
    assert args.info is True


def test_cli_main_info_execution(capsys):
    """Verify main function executes --info without failure and prints banner."""
    exit_code = main(["--info"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "MachSense" in captured.out
    assert "Project Root:" in captured.out
    assert "Target Column:" in captured.out


def test_cli_main_empty_args_shows_info(capsys):
    """Verify running CLI with empty args defaults to showing info."""
    exit_code = main([])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "MachSense" in captured.out


def test_cli_parser_explain_flag():
    """Verify CLI parser accepts --explain flag."""
    parser = build_parser()
    args = parser.parse_args(["--explain"])
    assert args.explain is True


def test_cli_main_explain_execution(capsys):
    """Verify running CLI with --explain executes successfully."""
    exit_code = main(["--explain"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "MachSense Explainable AI (SHAP) Demonstration" in captured.out
    assert "Computing Global Feature Importance" in captured.out


def test_cli_parser_predict_sample_flag():
    """Verify CLI parser accepts --predict-sample flag."""
    parser = build_parser()
    args = parser.parse_args(["--predict-sample"])
    assert args.predict_sample is True


def test_cli_main_predict_sample_execution(capsys):
    """Verify running CLI with --predict-sample executes successfully."""
    exit_code = main(["--predict-sample"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "MachSense Production Prediction Result" in captured.out
    assert "Failure Probability:" in captured.out
    assert "Operator Diagnostic Summary" in captured.out


def test_cli_parser_serve_flag():
    """Verify CLI parser accepts --serve, --host, and --port flags."""
    parser = build_parser()
    args = parser.parse_args(["--serve", "--host", "0.0.0.0", "--port", "9000"])
    assert args.serve is True
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_cli_parser_ui_flag():
    """Verify CLI parser accepts --ui flag."""
    parser = build_parser()
    args = parser.parse_args(["--ui"])
    assert args.ui is True
