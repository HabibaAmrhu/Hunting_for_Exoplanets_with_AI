#!/bin/bash
# Entrypoint script for production deployment
# Handles different service modes and initialization

set -e

# Default values
SERVICE_MODE=${1:-api}
LOG_LEVEL=${LOG_LEVEL:-INFO}
WORKERS=${WORKERS:-4}
PORT=${PORT:-8000}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] $1"
}

# Initialize application
initialize_app() {
    log "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/tmp
    
    # Set permissions
    chmod 755 /app/logs /app/tmp
    
    # Initialize database if needed
    if [ "$INIT_DB" = "true" ]; then
        log "Initializing database..."
        python -m src.database.init_db
    fi
    
    # Download models if needed
    if [ "$DOWNLOAD_MODELS" = "true" ]; then
        log "Downloading pre-trained models..."
        python -m src.models.download_pretrained
    fi
    
    log "Application initialized successfully"
}

# Health check function
health_check() {
    log "Running health check..."
    python -c "
import sys
import requests
try:
    response = requests.get('http://localhost:$PORT/health', timeout=5)
    if response.status_code == 200:
        print('Health check passed')
        sys.exit(0)
    else:
        print(f'Health check failed: {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'Health check error: {e}')
    sys.exit(1)
"
}

# Start API server
start_api() {
    log "Starting API server on port $PORT with $WORKERS workers"
    
    # Use gunicorn for production
    exec gunicorn \
        --bind 0.0.0.0:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 30 \
        --keep-alive 2 \
        --log-level $LOG_LEVEL \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --capture-output \
        --enable-stdio-inheritance \
        src.api.server:app
}

# Start worker process
start_worker() {
    log "Starting background worker"
    exec python -m src.worker.main
}

# Start scheduler
start_scheduler() {
    log "Starting task scheduler"
    exec python -m src.scheduler.main
}

# Start monitoring
start_monitoring() {
    log "Starting monitoring service"
    exec python -m src.monitoring.service
}

# Start Streamlit app
start_streamlit() {
    log "Starting Streamlit application on port $PORT"
    exec streamlit run streamlit_app/main.py \
        --server.port $PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
}

# Run training job
run_training() {
    log "Starting training job"
    exec python -m src.training.job "$@"
}

# Run inference job
run_inference() {
    log "Starting inference job"
    exec python -m src.inference.job "$@"
}

# Run data processing job
run_data_processing() {
    log "Starting data processing job"
    exec python -m src.data.processing_job "$@"
}

# Main execution
main() {
    log "Starting exoplanet detection pipeline"
    log "Service mode: $SERVICE_MODE"
    log "Log level: $LOG_LEVEL"
    
    # Initialize application
    initialize_app
    
    # Route to appropriate service
    case "$SERVICE_MODE" in
        "api")
            start_api
            ;;
        "worker")
            start_worker
            ;;
        "scheduler")
            start_scheduler
            ;;
        "monitoring")
            start_monitoring
            ;;
        "streamlit")
            start_streamlit
            ;;
        "training")
            shift
            run_training "$@"
            ;;
        "inference")
            shift
            run_inference "$@"
            ;;
        "data-processing")
            shift
            run_data_processing "$@"
            ;;
        "health-check")
            health_check
            ;;
        *)
            log "Unknown service mode: $SERVICE_MODE"
            log "Available modes: api, worker, scheduler, monitoring, streamlit, training, inference, data-processing, health-check"
            exit 1
            ;;
    esac
}

# Handle signals for graceful shutdown
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' TERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' INT

# Execute main function
main "$@"