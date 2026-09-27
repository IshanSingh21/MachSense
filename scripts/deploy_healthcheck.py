"""Deployment health check and live verification utility for MachSense."""

from __future__ import annotations

import argparse
import sys
import time

try:
    import httpx
except ImportError:
    print("Error: 'httpx' is required. Install it using 'pip install httpx'.")
    sys.exit(1)


def check_deployment(base_url: str, timeout: float = 10.0) -> bool:
    """Run full synthetic health check and sample inference on a deployed instance."""
    base_url = base_url.rstrip("/")
    print("=" * 70)
    print(f"MachSense Live Deployment Health Check: {base_url}")
    print("=" * 70)

    client = httpx.Client(timeout=timeout)

    # 1. Test Root / Discovery
    try:
        t0 = time.perf_counter()
        resp = client.get(f"{base_url}/")
        dt_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            print(f" [PASS] Root Discovery Endpoint ({resp.status_code}) - {dt_ms:.1f}ms")
            print(f"        Response: {resp.json().get('message')}")
        else:
            print(f" [WARN] Root Discovery returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f" [FAIL] Could not connect to Root endpoint: {e}")
        return False

    # 2. Test Liveness Probe (/health)
    try:
        t0 = time.perf_counter()
        resp = client.get(f"{base_url}/health")
        dt_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200 and resp.json().get("status") == "healthy":
            print(f" [PASS] Liveness Probe /health ({resp.status_code}) - {dt_ms:.1f}ms")
            print(f"        Environment: {resp.json().get('environment')}, Version: {resp.json().get('version')}")
        else:
            print(f" [FAIL] Liveness Probe failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f" [FAIL] Liveness Probe error: {e}")
        return False

    # 3. Test Readiness Probe (/health/ready)
    try:
        t0 = time.perf_counter()
        resp = client.get(f"{base_url}/health/ready")
        dt_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200 and resp.json().get("status") == "ready":
            data = resp.json()
            print(f" [PASS] Readiness Probe /health/ready ({resp.status_code}) - {dt_ms:.1f}ms")
            print(f"        Model: {data.get('model_name')} (v{data.get('model_version')}) | Optimal Threshold: {data.get('optimal_threshold')}")
        else:
            print(f" [FAIL] Readiness Probe failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f" [FAIL] Readiness Probe error: {e}")
        return False

    # 4. Test Live Single Prediction (/api/v1/predict)
    payload = {
        "telemetry": {
            "type": "L",
            "air_temperature_k": 302.5,
            "process_temperature_k": 311.8,
            "rotational_speed_rpm": 1380.0,
            "torque_nm": 62.5,
            "tool_wear_min": 215.0,
        },
        "explain": True,
    }

    try:
        t0 = time.perf_counter()
        resp = client.post(f"{base_url}/api/v1/predict", json=payload)
        dt_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            pred = resp.json()
            print(f" [PASS] Live Prediction Scoring ({resp.status_code}) - {dt_ms:.1f}ms")
            print(f"        Prediction: {pred.get('predicted_label')} (Risk: {pred.get('risk_level')})")
            print(f"        Failure Prob: {pred.get('failure_probability'):.2%} | Model Latency: {pred.get('latency_ms')}ms")
            if pred.get("top_risk_escalators"):
                top_feature = pred["top_risk_escalators"][0]
                print(f"        Top Escalator: {top_feature['feature_name']} (SHAP: +{top_feature['attribution_value']:.4f})")
        else:
            print(f" [FAIL] Prediction request failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f" [FAIL] Prediction error: {e}")
        return False

    print("=" * 70)
    print(" ALL DEPLOYMENT HEALTH CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MachSense Deployment Healthcheck")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of deployed API (e.g. http://127.0.0.1:8000 or https://machsense-api.onrender.com)",
    )
    args = parser.parse_args()
    success = check_deployment(args.url)
    sys.exit(0 if success else 1)
