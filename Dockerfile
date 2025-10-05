# Multi-stage Docker build for exoplanet detection pipeline
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash exoplanet
WORKDIR /home/exoplanet

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir -r requirements-dev.txt
USER exoplanet
COPY --chown=exoplanet:exoplanet . .
CMD ["python", "-m", "streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy application code
COPY --chown=exoplanet:exoplanet src/ ./src/
COPY --chown=exoplanet:exoplanet streamlit_app/ ./streamlit_app/
COPY --chown=exoplanet:exoplanet scripts/ ./scripts/
COPY --chown=exoplanet:exoplanet models/ ./models/
COPY --chown=exoplanet:exoplanet run_quick_test.py ./

# Create necessary directories
RUN mkdir -p data results logs && \
    chown -R exoplanet:exoplanet data results logs

# Switch to non-root user
USER exoplanet

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Default command
CMD ["python", "-m", "streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]