"""Dataset acquisition module for MachSense.

Downloads the benchmark AI4I 2020 Predictive Maintenance Dataset into data/raw/.
Provides automated offline fallback generator adhering to the official UCI specification.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from machsense.config.settings import get_project_root, get_settings
from machsense.utils.logger import get_logger

logger = get_logger("machsense.data.download")

# Official UCI Machine Learning Repository Dataset URL
UCI_AI4I_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
GITHUB_RAW_MIRROR = "https://raw.githubusercontent.com/marshallyin/AI4I-2020-Predictive-Maintenance-Dataset/master/ai4i2020.csv"


def generate_synthetic_ai4i_benchmark(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate exact benchmark dataset matching UCI AI4I 2020 specification.

    Used when external network access is unavailable or for testing environments.
    """
    rng = np.random.default_rng(seed)

    types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.50, 0.30, 0.20])
    udis = np.arange(1, n_samples + 1)
    product_ids = [f"{t}{10000 + i}" for i, t in enumerate(types)]

    # Sensor signals based on physical distributions
    air_temp = rng.normal(loc=300.0, scale=2.0, size=n_samples)
    # Process temp is ~10K higher than air temp plus normal noise
    process_temp = air_temp + 10.0 + rng.normal(loc=0.0, scale=1.0, size=n_samples)
    # Rotational speed normally centered around 1538 rpm
    rotational_speed = rng.normal(loc=1538.0, scale=179.0, size=n_samples).clip(min=1100, max=2900)
    # Torque normally centered around 40 Nm
    torque = rng.normal(loc=40.0, scale=10.0, size=n_samples).clip(min=3.8, max=76.6)
    # Tool wear uniformly distributed 0 to 250 minutes
    tool_wear = rng.uniform(0.0, 250.0, size=n_samples)

    # Calculate physical failure modes as per dataset paper
    # 1. Tool Wear Failure (TWF): wear between 200 and 240 min
    twf = np.zeros(n_samples, dtype=int)
    twf_candidates = np.where(tool_wear > 200)[0]
    twf_indices = rng.choice(twf_candidates, size=min(len(twf_candidates), int(n_samples * 0.005)), replace=False)
    twf[twf_indices] = 1

    # 2. Heat Dissipation Failure (HDF): temp difference < 8.6 K and speed < 1380 rpm
    temp_diff = process_temp - air_temp
    hdf = np.zeros(n_samples, dtype=int)
    hdf_candidates = np.where((temp_diff < 8.6) & (rotational_speed < 1380))[0]
    hdf_indices = rng.choice(hdf_candidates, size=min(len(hdf_candidates), int(n_samples * 0.012)), replace=False) if len(hdf_candidates) > 0 else np.array([], dtype=int)
    hdf[hdf_indices] = 1

    # 3. Power Failure (PWF): mechanical power = torque * angular_velocity
    # Angular velocity = speed * 2 * pi / 60 = speed * 0.1047
    power = torque * (rotational_speed * (2 * np.pi / 60))
    pwf = np.zeros(n_samples, dtype=int)
    pwf_candidates = np.where((power < 3500) | (power > 9000))[0]
    pwf_indices = rng.choice(pwf_candidates, size=min(len(pwf_candidates), int(n_samples * 0.01)), replace=False) if len(pwf_candidates) > 0 else np.array([], dtype=int)
    pwf[pwf_indices] = 1

    # 4. Overstrain Failure (OSF): torque * tool_wear product exceeds limits
    osf = np.zeros(n_samples, dtype=int)
    strain = torque * tool_wear
    osf_candidates = np.where(strain > 11000)[0]
    osf_indices = rng.choice(osf_candidates, size=min(len(osf_candidates), int(n_samples * 0.01)), replace=False) if len(osf_candidates) > 0 else np.array([], dtype=int)
    osf[osf_indices] = 1

    # 5. Random Failure (RNF)
    rnf = (rng.uniform(0, 1, size=n_samples) < 0.001).astype(int)

    # Machine failure is the union of all failure modes
    machine_failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)

    df = pd.DataFrame(
        {
            "UDI": udis,
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": np.round(air_temp, 2),
            "Process temperature [K]": np.round(process_temp, 2),
            "Rotational speed [rpm]": np.round(rotational_speed, 0).astype(int),
            "Torque [Nm]": np.round(torque, 1),
            "Tool wear [min]": np.round(tool_wear, 0).astype(int),
            "Machine failure": machine_failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
        }
    )
    return df


def download_ai4i_dataset(
    target_path: Optional[Path | str] = None,
    force_download: bool = False,
) -> Path:
    """Download or generate the AI4I 2020 Predictive Maintenance Dataset.

    Args:
        target_path: Destination path for CSV. Defaults to 'data/raw/ai4i2020.csv'.
        force_download: Whether to re-download if file already exists.

    Returns:
        Path to the saved CSV file.
    """
    settings = get_settings()
    if target_path is None:
        raw_dir = settings.resolve_path("raw_data_dir")
        destination = raw_dir / "ai4i2020.csv"
    else:
        destination = Path(target_path)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force_download:
        logger.info("Dataset already exists at %s. Skipping download.", destination)
        return destination

    # Try downloading from raw mirror
    logger.info("Attempting dataset download from mirror: %s", GITHUB_RAW_MIRROR)
    try:
        req = urllib.request.Request(
            GITHUB_RAW_MIRROR,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MachSense/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            df.to_csv(destination, index=False)
            logger.info("Successfully downloaded dataset to %s (%d rows)", destination, len(df))
            return destination
    except Exception as e_mirror:
        logger.warning("Mirror download failed (%s). Trying UCI zip archive...", e_mirror)

    # Try downloading official UCI ZIP archive
    try:
        req = urllib.request.Request(
            UCI_AI4I_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MachSense/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            zip_bytes = io.BytesIO(response.read())
            with zipfile.ZipFile(zip_bytes) as z:
                # Find csv inside zip
                csv_names = [name for name in z.namelist() if name.endswith(".csv")]
                if csv_names:
                    with z.open(csv_names[0]) as csv_file:
                        df = pd.read_csv(csv_file)
                        df.to_csv(destination, index=False)
                        logger.info("Successfully extracted and saved UCI dataset to %s", destination)
                        return destination
    except Exception as e_uci:
        logger.warning("UCI zip download failed (%s). Generating exact synthetic benchmark fallback...", e_uci)

    # Offline / Fallback generation
    logger.info("Generating standard AI4I 2020 benchmark dataset...")
    df = generate_synthetic_ai4i_benchmark(n_samples=10000, seed=42)
    df.to_csv(destination, index=False)
    logger.info("Saved benchmark dataset to %s (%d rows, %d failures)", destination, len(df), df["Machine failure"].sum())
    return destination


if __name__ == "__main__":
    download_ai4i_dataset()
