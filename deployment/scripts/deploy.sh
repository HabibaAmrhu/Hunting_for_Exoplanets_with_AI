#!/bin/bash

# Production deployment script for exoplanet detection pipeline
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-latest}"
NAMESPACE="${NAMESPACE:-exoplanet-detection}"
REGISTRY="${REGISTRY:-your-registry.com}"
BUILD_PUSH="${BUILD_PUSH:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    if [[ "$BUILD_PUSH" != "true" ]]; then
        log_info "Skipping image build and push"
        return
    fi
    
    log_info "Building Docker image..."
    
    local image_tag="${REGISTRY}/exoplanet-detection:${VERSION}"
    local build_args=(
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg" "VERSION=${VERSION}"
        "--build-arg" "VCS_REF=$(git rev-parse --short HEAD)"
    )
    
    # Build image
    docker build \
        -f "${DEPLOYMENT_DIR}/docker/Dockerfile.production" \
        "${build_args[@]}" \
        -t "$image_tag" \
        "$PROJECT_ROOT"
    
    log_success "Docker image built: $image_tag"
    
    # Push image
    log_info "Pushing Docker image to registry..."
    docker push "$image_tag"
    log_success "Docker image pushed: $image_tag"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/namespace.yaml"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets exist
    if kubectl get secret exoplanet-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist. Skipping deployment."
        log_warning "To update secrets, delete them first: kubectl delete secret exoplanet-secrets -n $NAMESPACE"
    else
        kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/secrets.yaml"
        log_success "Secrets deployed"
    fi
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/configmap.yaml"
    log_success "ConfigMaps deployed"
}

# Deploy storage
deploy_storage() {
    log_info "Deploying storage resources..."
    kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/storage.yaml"
    log_success "Storage resources deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."
    
    # Update image tag in deployment
    local temp_deployment="/tmp/deployment-${VERSION}.yaml"
    sed "s|exoplanet-detection:latest|${REGISTRY}/exoplanet-detection:${VERSION}|g" \
        "${DEPLOYMENT_DIR}/kubernetes/deployment.yaml" > "$temp_deployment"
    
    kubectl apply -f "$temp_deployment"
    kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/service.yaml"
    
    rm -f "$temp_deployment"
    log_success "Application deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring resources..."
    kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/monitoring.yaml"
    log_success "Monitoring resources deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    local deployments=("exoplanet-api" "exoplanet-worker" "postgresql" "redis")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s \
            deployment/"$deployment" -n "$NAMESPACE" || {
            log_error "Deployment $deployment failed to become ready"
            return 1
        }
    done
    
    log_success "All deployments are ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service exoplanet-api-service -n "$NAMESPACE" \
        -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for health check
    kubectl port-forward service/exoplanet-api-service 8080:80 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    sleep 5
    
    # Health check
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    kill $port_forward_pid 2>/dev/null || true
    log_success "Health checks completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    kubectl rollout undo deployment/exoplanet-api -n "$NAMESPACE"
    kubectl rollout undo deployment/exoplanet-worker -n "$NAMESPACE"
    
    log_success "Rollback completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of exoplanet detection pipeline"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Namespace: $NAMESPACE"
    log_info "Registry: $REGISTRY"
    
    # Run deployment steps
    check_prerequisites
    build_and_push_image
    create_namespace
    deploy_secrets
    deploy_configmaps
    deploy_storage
    deploy_application
    deploy_monitoring
    
    # Wait and verify
    if wait_for_deployment && run_health_checks; then
        log_success "Deployment completed successfully!"
        
        # Display access information
        echo
        log_info "Access information:"
        echo "  API: kubectl port-forward service/exoplanet-api-service 8080:80 -n $NAMESPACE"
        echo "  Grafana: kubectl port-forward service/grafana 3000:3000 -n $NAMESPACE"
        echo "  Prometheus: kubectl port-forward service/prometheus 9090:9090 -n $NAMESPACE"
        
    else
        log_error "Deployment failed!"
        log_warning "Consider running rollback: $0 rollback"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [deploy|rollback|status|logs]"
    echo
    echo "Commands:"
    echo "  deploy    - Deploy the application (default)"
    echo "  rollback  - Rollback to previous version"
    echo "  status    - Show deployment status"
    echo "  logs      - Show application logs"
    echo
    echo "Environment variables:"
    echo "  ENVIRONMENT - Deployment environment (default: production)"
    echo "  VERSION     - Image version tag (default: latest)"
    echo "  NAMESPACE   - Kubernetes namespace (default: exoplanet-detection)"
    echo "  REGISTRY    - Docker registry (default: your-registry.com)"
    echo "  BUILD_PUSH  - Build and push image (default: true)"
}

# Show deployment status
show_status() {
    log_info "Deployment status for namespace: $NAMESPACE"
    
    echo
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo
    echo "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE"
    
    echo
    echo "PVCs:"
    kubectl get pvc -n "$NAMESPACE"
}

# Show logs
show_logs() {
    local pod_name="${1:-}"
    
    if [[ -z "$pod_name" ]]; then
        log_info "Available pods:"
        kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name"
        echo
        log_info "Usage: $0 logs <pod-name>"
        return
    fi
    
    log_info "Showing logs for pod: $pod_name"
    kubectl logs -f "$pod_name" -n "$NAMESPACE"
}

# Main script logic
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            deploy
            ;;
        "rollback")
            rollback_deployment
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"