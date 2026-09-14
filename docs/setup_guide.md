# MachSense Developer Setup Guide

This guide walks through setting up the local environment, installing dependencies, and running tests.

## 1. Prerequisites
- Python 3.10 or higher (Python 3.13 tested)
- pip or conda package manager
- Git

## 2. Environment Setup

### Option A: Standard `venv`
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source .venv/bin/activate
```

### Option B: Conda
```bash
conda create -n machsense python=3.11 -y
conda activate machsense
```

## 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development & test dependencies
pip install -r requirements-dev.txt

# Install machsense in editable mode
pip install -e .
```

## 4. Verify Installation & Configuration
Run the CLI info check:
```bash
python -m machsense.main --info
```

Run unit tests:
```bash
pytest
```

## 5. Configuration Overrides
Configurations are loaded from `configs/config.yaml`. You can override parameters using environment variables:
- `MACHSENSE_ROOT`: Custom project root directory.
- `MACHSENSE_CONFIG_PATH`: Path to an alternate YAML config file.
- `MACHSENSE_ENV`: Set runtime environment (e.g. `production`, `staging`, `testing`).
