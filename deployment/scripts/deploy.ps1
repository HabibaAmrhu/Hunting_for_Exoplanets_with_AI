# PowerShell deployment script for Windows
# Production-ready deployment pipeline for exoplanet detection system

param(
    [Parameter(Mandatory=$true)]
    [string]$Environment,
    
    [Parameter(Mandatory=$false)]
    [string]$Version = "latest",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Configuration
$ErrorActionPreference = "Stop"
$ProjectName = "exoplanet-detection-pipeline"
$DockerRegistry = "your-registry.com"
$KubernetesNamespace = "exoplanet-$Environment"

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $(
        switch ($Level) {
            "ERROR" { "Red" }
            "WARN" { "Yellow" }
            "SUCCESS" { "Green" }
            default { "White" }
        }
    )
}

# Validate environment
function Test-Environment {
    param([string]$Env)
    
    $validEnvironments = @("dev", "staging", "prod")
    if ($Env -notin $validEnvironments) {
        Write-Log "Invalid environment: $Env. Valid options: $($validEnvironments -join ', ')" "ERROR"
        exit 1
    }
    
    Write-Log "Deploying to environment: $Env" "SUCCESS"
}

# Check prerequisites
function Test-Prerequisites {
    Write-Log "Checking prerequisites..."
    
    # Check Docker
    try {
        docker --version | Out-Null
        Write-Log "Docker is available" "SUCCESS"
    } catch {
        Write-Log "Docker is not installed or not in PATH" "ERROR"
        exit 1
    }
    
    # Check kubectl (for Kubernetes deployments)
    try {
        kubectl version --client | Out-Null
        Write-Log "kubectl is available" "SUCCESS"
    } catch {
        Write-Log "kubectl is not installed or not in PATH" "WARN"
    }
    
    # Check Python
    try {
        python --version | Out-Null
        Write-Log "Python is available" "SUCCESS"
    } catch {
        Write-Log "Python is not installed or not in PATH" "ERROR"
        exit 1
    }
}

# Run tests
function Invoke-Tests {
    if ($SkipTests) {
        Write-Log "Skipping tests as requested" "WARN"
        return
    }
    
    Write-Log "Running test suite..."
    
    # Install test dependencies
    pip install -r requirements-dev.txt
    
    # Run linting
    Write-Log "Running code quality checks..."
    flake8 src/
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Linting failed" "ERROR"
        exit 1
    }
    
    # Run tests
    Write-Log "Running unit tests..."
    pytest tests/ --cov=src/ --cov-report=xml --junitxml=test-results.xml
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Tests failed" "ERROR"
        exit 1
    }
    
    Write-Log "All tests passed" "SUCCESS"
}

# Build Docker image
function Build-DockerImage {
    param([string]$Tag)
    
    Write-Log "Building Docker image with tag: $Tag"
    
    # Build production image
    docker build -f deployment/docker/Dockerfile.prod -t "${ProjectName}:${Tag}" .
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Docker build failed" "ERROR"
        exit 1
    }
    
    # Tag for registry
    docker tag "${ProjectName}:${Tag}" "${DockerRegistry}/${ProjectName}:${Tag}"
    
    Write-Log "Docker image built successfully" "SUCCESS"
}

# Push Docker image
function Push-DockerImage {
    param([string]$Tag)
    
    Write-Log "Pushing Docker image to registry..."
    
    # Login to registry (assumes credentials are configured)
    docker push "${DockerRegistry}/${ProjectName}:${Tag}"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Docker push failed" "ERROR"
        exit 1
    }
    
    Write-Log "Docker image pushed successfully" "SUCCESS"
}

# Deploy to Kubernetes
function Deploy-ToKubernetes {
    param([string]$Env, [string]$Tag)
    
    Write-Log "Deploying to Kubernetes environment: $Env"
    
    # Create namespace if it doesn't exist
    kubectl create namespace $KubernetesNamespace --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configuration
    $configFiles = @(
        "deployment/kubernetes/configmap-$Env.yaml",
        "deployment/kubernetes/secret-$Env.yaml",
        "deployment/kubernetes/deployment.yaml",
        "deployment/kubernetes/service.yaml",
        "deployment/kubernetes/ingress-$Env.yaml"
    )
    
    foreach ($configFile in $configFiles) {
        if (Test-Path $configFile) {
            Write-Log "Applying $configFile"
            
            # Replace image tag in deployment
            if ($configFile -like "*deployment.yaml") {
                (Get-Content $configFile) -replace 'IMAGE_TAG_PLACEHOLDER', $Tag | kubectl apply -n $KubernetesNamespace -f -
            } else {
                kubectl apply -n $KubernetesNamespace -f $configFile
            }
            
            if ($LASTEXITCODE -ne 0) {
                Write-Log "Failed to apply $configFile" "ERROR"
                exit 1
            }
        } else {
            Write-Log "Configuration file not found: $configFile" "WARN"
        }
    }
    
    Write-Log "Kubernetes deployment completed" "SUCCESS"
}

# Wait for deployment
function Wait-ForDeployment {
    param([string]$Env)
    
    Write-Log "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/exoplanet-pipeline -n $KubernetesNamespace
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Deployment failed to become ready" "ERROR"
        exit 1
    }
    
    Write-Log "Deployment is ready" "SUCCESS"
}

# Run health checks
function Test-HealthChecks {
    param([string]$Env)
    
    Write-Log "Running health checks..."
    
    # Get service URL
    if ($Env -eq "prod") {
        $serviceUrl = "https://exoplanet-api.yourdomain.com"
    } else {
        $serviceUrl = "https://exoplanet-api-$Env.yourdomain.com"
    }
    
    # Health check endpoint
    try {
        $response = Invoke-RestMethod -Uri "$serviceUrl/health" -Method Get -TimeoutSec 30
        if ($response.status -eq "healthy") {
            Write-Log "Health check passed" "SUCCESS"
        } else {
            Write-Log "Health check failed: $($response.status)" "ERROR"
            exit 1
        }
    } catch {
        Write-Log "Health check failed: $($_.Exception.Message)" "ERROR"
        exit 1
    }
}

# Rollback deployment
function Invoke-Rollback {
    param([string]$Env)
    
    Write-Log "Rolling back deployment..." "WARN"
    
    kubectl rollout undo deployment/exoplanet-pipeline -n $KubernetesNamespace
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Rollback failed" "ERROR"
        exit 1
    }
    
    Write-Log "Rollback completed" "SUCCESS"
}

# Cleanup old images
function Remove-OldImages {
    Write-Log "Cleaning up old Docker images..."
    
    # Remove old local images (keep last 5)
    $images = docker images "${ProjectName}" --format "table {{.Tag}}" | Select-Object -Skip 1 | Select-Object -Skip 5
    foreach ($image in $images) {
        if ($image -and $image -ne "latest") {
            docker rmi "${ProjectName}:${image}" -f
        }
    }
    
    Write-Log "Cleanup completed" "SUCCESS"
}

# Main deployment function
function Start-Deployment {
    param([string]$Env, [string]$Ver)
    
    Write-Log "Starting deployment pipeline for $ProjectName"
    Write-Log "Environment: $Env, Version: $Ver"
    
    try {
        # Validate inputs
        Test-Environment -Env $Env
        
        # Check prerequisites
        Test-Prerequisites
        
        # Run tests
        Invoke-Tests
        
        # Build and push image
        $imageTag = if ($Ver -eq "latest") { "$(Get-Date -Format 'yyyyMMdd-HHmmss')" } else { $Ver }
        Build-DockerImage -Tag $imageTag
        Push-DockerImage -Tag $imageTag
        
        # Deploy to Kubernetes
        Deploy-ToKubernetes -Env $Env -Tag $imageTag
        
        # Wait for deployment
        Wait-ForDeployment -Env $Env
        
        # Run health checks
        Test-HealthChecks -Env $Env
        
        # Cleanup
        Remove-OldImages
        
        Write-Log "Deployment completed successfully!" "SUCCESS"
        Write-Log "Application is available at: https://exoplanet-api$(if($Env -ne 'prod'){'-' + $Env}).yourdomain.com" "SUCCESS"
        
    } catch {
        Write-Log "Deployment failed: $($_.Exception.Message)" "ERROR"
        
        if (-not $Force) {
            $rollback = Read-Host "Do you want to rollback? (y/N)"
            if ($rollback -eq 'y' -or $rollback -eq 'Y') {
                Invoke-Rollback -Env $Env
            }
        }
        
        exit 1
    }
}

# Script execution
Write-Log "Exoplanet Detection Pipeline Deployment Script"
Write-Log "=============================================="

Start-Deployment -Env $Environment -Ver $Version