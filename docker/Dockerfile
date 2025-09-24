# Multi-stage Docker build for Advanced Trading System
# Optimized for production deployment with security and performance

# Stage 1: Base Python environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trader
USER trader
WORKDIR /home/trader

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Stage 3: Application code
FROM dependencies as application

# Copy application code
COPY --chown=trader:trader . /home/trader/trading-system/

# Set working directory
WORKDIR /home/trader/trading-system

# Install the trading system package
RUN pip install --user -e .

# Stage 4: Production image
FROM application as production

# Expose ports
EXPOSE 8000 8501 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "core.main"]

# Development stage
FROM application as development

# Install development dependencies
RUN pip install --user -r requirements-dev.txt

# Install additional development tools
RUN pip install --user \
    jupyter \
    jupyterlab \
    black \
    flake8 \
    mypy \
    pre-commit

# Expose additional ports for development
EXPOSE 8888 5000

# Development command
CMD ["bash"]