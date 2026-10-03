"""Unit and integration tests for deployment configurations and healthchecks."""


from fastapi.testclient import TestClient

from machsense.api.app import create_app
from machsense.config.settings import get_project_root


def test_deployment_files_exist():
    """Verify all critical deployment configuration files exist."""
    root = get_project_root()
    expected_files = [
        "Dockerfile",
        "Dockerfile.api",
        "frontend/Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "render.yaml",
        "Procfile",
        "scripts/deploy_healthcheck.py",
        "requirements.txt",
    ]
    for rel_path in expected_files:
        p = root / rel_path
        assert p.exists(), f"Missing required deployment artifact: {rel_path}"
        assert p.stat().st_size > 0, f"Deployment artifact is empty: {rel_path}"


def test_dockerfile_contents():
    """Verify Dockerfiles specify secure non-root users, healthchecks, and ports."""
    root = get_project_root()
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "useradd" in dockerfile
    assert "USER machsense" in dockerfile
    assert "EXPOSE" in dockerfile


def test_docker_compose_structure():
    """Verify docker-compose.yml contains both api and frontend services with healthchecks."""
    root = get_project_root()
    import yaml

    with open(root / "docker-compose.yml", "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    assert "services" in compose
    assert "api" in compose["services"]
    assert "frontend" in compose["services"]
    assert "healthcheck" in compose["services"]["api"]


def test_liveness_health_live_endpoint():
    """Verify /health/live returns healthy status for orchestrator probes."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
