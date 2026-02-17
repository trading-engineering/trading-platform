# syntax=docker/dockerfile:1.4

FROM python:3.11.14-slim-trixie

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

# Set working directory for trading-platform
WORKDIR /workspace/trading-platform

# Default command for the devcontainer (keeps container alive)
CMD ["sleep", "infinity"]
