# ==============================================================================
# MachSense FastAPI Backend Production Container
# Supports: FastAPI Predictive Maintenance Engine (--serve)
# ==============================================================================

FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:${PYTHONPATH}" \
    MACHSENSE_ENV=production \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install security updates and curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and model artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY app/ ./app/
COPY pyproject.toml .
COPY README.md .

# Create non-root user and setup permissions
RUN useradd -m -u 1000 machsense && \
    mkdir -p /app/logs && \
    chown -R machsense:machsense /app

USER machsense

# Expose default API port
EXPOSE 8000

# Default Container Healthcheck (Liveness Probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default launch command: FastAPI production server
CMD ["uvicorn", "machsense.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
