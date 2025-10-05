# Developer Guide

## Exoplanet Detection Pipeline - Developer Guide

This guide provides comprehensive information for developers working on the exoplanet detection pipeline.

### Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Organization](#code-organization)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Performance Guidelines](#performance-guidelines)
8. [Security Considerations](#security-considerations)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

---

## Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Processing     │    │   Model Layer   │
│                 │    │     Layer       │    │                 │
│ • Raw Data      │───▶│ • Preprocessing │───▶│ • CNN Models    │
│ • Processed     │    │ • Augmentation  │    │ • Transformers  │
│ • Synthetic     │    │ • Validation    │    │ • Ensembles     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Storage Layer  │    │  Training       │    │  Inference      │
│                 │    │     Layer       │    │     Layer       │
│ • File System   │    │ • Trainers      │    │ • Batch Proc.   │
│ • Databases     │    │ • Metrics       │    │ • Real-time     │
│ • Cloud Storage │    │ • Optimization  │    │ • API Endpoints │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Application    │
                    │     Layer       │
                    │ • Web Interface │
                    │ • CLI Tools     │
                    │ • Monitoring    │
                    └─────────────────┘
```

### Core Components

1. **Data Management**: Handles data ingestion, preprocessing, and storage
2. **Model Framework**: Provides model architectures and training utilities
3. **Training Pipeline**: Manages model training, validation, and optimization
4. **Inference Engine**: Handles model deployment and prediction serving
5. **Monitoring System**: Tracks performance, drift, and system health
6. **Security Layer**: Manages authentication, authorization, and data protection

### Design Principles

- **Modularity**: Each component is self-contained and reusable
- **Extensibility**: Easy to add new models, data sources, and features
- **Scalability**: Designed to handle large datasets and high throughput
- **Reliability**: Robust error handling and recovery mechanisms
- **Security**: Built-in security features and best practices

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- CUDA-capable GPU (optional)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
pre-commit>=2.20.0
sphinx>=5.0.0
jupyter>=1.0.0
tensorboard>=2.10.0
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Configuration

#### VSCode Settings

```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

---

## Code Organization

### Directory Structure

```
exoplanet-detection-pipeline/
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset classes
│   │   ├── augmentation.py      # Data augmentation
│   │   ├── downloader.py        # Data download utilities
│   │   └── types.py             # Data type definitions
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── base.py              # Base model classes
│   │   ├── cnn.py               # CNN implementations
│   │   ├── transformer.py       # Transformer models
│   │   └── ensemble.py          # Ensemble methods
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loops
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── callbacks.py         # Training callbacks
│   ├── inference/                # Inference utilities
│   │   ├── __init__.py
│   │   ├── predictor.py         # Prediction interface
│   │   └── batch_processor.py   # Batch processing
│   ├── preprocessing/            # Data preprocessing
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Main preprocessor
│   │   └── pipeline.py          # Processing pipelines
│   ├── explainability/          # Model interpretability
│   │   ├── __init__.py
│   │   └── explainer.py         # Explanation methods
│   ├── monitoring/               # System monitoring
│   │   ├── __init__.py
│   │   └── monitor.py           # Performance monitoring
│   ├── security/                 # Security features
│   │   ├── __init__.py
│   │   └── auth.py              # Authentication
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   ├── server.py            # API server
│   │   └── endpoints.py         # Route definitions
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   └── logging.py           # Logging utilities
│   └── visualization/            # Plotting and visualization
│       ├── __init__.py
│       └── plots.py             # Plotting functions
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   ├── test_data/               # Test data processing
│   ├── test_models/             # Test model functionality
│   ├── test_training/           # Test training pipeline
│   └── test_integration/        # Integration tests
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── scripts/                      # Utility scripts
│   ├── train_baseline.py
│   ├── evaluate_models.py
│   └── download_data.py
├── docs/                         # Documentation
│   ├── api_reference.md
│   ├── user_guide.md
│   └── developer_guide.md
├── deployment/                   # Deployment configurations
│   ├── docker/
│   ├── kubernetes/
│   └── scripts/
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project configuration
└── README.md                     # Project overview
```

### Coding Standards

#### Python Style Guide

Follow PEP 8 with these additions:

```python
# Import organization
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.utils.config import Config

# Type hints
from typing import Dict, List, Optional, Tuple, Union

def process_data(
    data: np.ndarray,
    config: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process input data according to configuration.
    
    Args:
        data: Input data array
        config: Processing configuration
        output_path: Optional output file path
        
    Returns:
        Tuple of processed data and metrics
        
    Raises:
        ValueError: If data format is invalid
    """
    pass
```

#### Naming Conventions

- **Classes**: PascalCase (`ExoplanetCNN`, `DataProcessor`)
- **Functions/Variables**: snake_case (`train_model`, `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_EPOCHS`, `DEFAULT_BATCH_SIZE`)
- **Private members**: Leading underscore (`_internal_method`)

#### Documentation Standards

```python
class ExoplanetModel(nn.Module):
    """
    Base class for exoplanet detection models.
    
    This class provides common functionality for all exoplanet detection
    models including data preprocessing, training utilities, and evaluation
    methods.
    
    Attributes:
        input_dim: Dimension of input features
        output_dim: Dimension of output predictions
        device: Device for model computations
        
    Example:
        >>> model = ExoplanetModel(input_dim=1000)
        >>> predictions = model(data)
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            device: Device for computations ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = torch.device(device)
```

---

## Development Workflow

### Git Workflow

We use GitFlow with the following branches:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `release/*`: Release preparation branches
- `hotfix/*`: Critical bug fixes

#### Feature Development

```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/new-model-architecture

# Make changes and commit
git add .
git commit -m "feat: add transformer model architecture"

# Push and create pull request
git push origin feature/new-model-architecture
```

#### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Code Review Process

1. **Create Pull Request**: Include description and testing notes
2. **Automated Checks**: Ensure all CI checks pass
3. **Peer Review**: At least one reviewer approval required
4. **Testing**: Verify functionality works as expected
5. **Merge**: Squash and merge to maintain clean history

### Continuous Integration

#### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        mypy src/
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src/ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Testing

### Testing Strategy

We use a multi-layered testing approach:

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

### Test Organization

```python
# tests/test_models/test_cnn.py
import pytest
import torch
from src.models import ExoplanetCNN

class TestExoplanetCNN:
    """Test suite for ExoplanetCNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return ExoplanetCNN(input_length=1000)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return torch.randn(32, 1, 1000)
    
    def test_model_creation(self, model):
        """Test model can be created successfully."""
        assert isinstance(model, ExoplanetCNN)
        assert model.input_length == 1000
    
    def test_forward_pass(self, model, sample_data):
        """Test model forward pass."""
        model.eval()
        with torch.no_grad():
            output = model(sample_data)
        
        assert output.shape == (32, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_training_mode(self, model, sample_data):
        """Test model in training mode."""
        model.train()
        output = model(sample_data)
        
        # Test gradient computation
        loss = torch.nn.BCELoss()(output, torch.ones_like(output))
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model with different batch sizes."""
        data = torch.randn(batch_size, 1, 1000)
        output = model(data)
        assert output.shape == (batch_size, 1)
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_light_curve():
    """Generate sample light curve data."""
    time = np.linspace(0, 100, 1000)
    flux = np.ones_like(time) + 0.01 * np.random.randn(len(time))
    return {'time': time, 'flux': flux}

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ --cov-report=html

# Run specific test file
pytest tests/test_models/test_cnn.py

# Run tests matching pattern
pytest -k "test_model"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Performance Testing

```python
# tests/test_performance.py
import time
import pytest
import torch
from src.models import ExoplanetCNN

class TestPerformance:
    """Performance tests for critical components."""
    
    @pytest.mark.performance
    def test_inference_speed(self):
        """Test model inference speed."""
        model = ExoplanetCNN(input_length=1000)
        model.eval()
        
        data = torch.randn(100, 1, 1000)
        
        # Warmup
        with torch.no_grad():
            _ = model(data[:10])
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            predictions = model(data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        samples_per_second = len(data) / inference_time
        
        # Assert minimum performance
        assert samples_per_second > 100, f"Too slow: {samples_per_second:.1f} samples/sec"
```

---

## Documentation

### Documentation Standards

All public APIs must be documented with:

1. **Purpose**: What the function/class does
2. **Parameters**: All parameters with types and descriptions
3. **Returns**: Return values with types and descriptions
4. **Raises**: Exceptions that may be raised
5. **Examples**: Usage examples when helpful

### Sphinx Documentation

```python
# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

html_theme = 'sphinx_rtd_theme'
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### API Documentation Generation

```bash
# Generate API documentation
sphinx-apidoc -o docs/api src/

# Update documentation
cd docs/
make clean
make html
```

---

## Performance Guidelines

### Memory Management

```python
# Efficient data loading
class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        # Don't load all data at once
    
    def __getitem__(self, idx):
        # Load data on demand
        data = np.load(self.data_paths[idx])
        return torch.from_numpy(data)

# Memory-efficient training
def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for i, (data, targets) in enumerate(dataloader):
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### GPU Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step_mixed_precision(model, data, targets, optimizer, criterion):
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# Efficient data transfer
def efficient_data_transfer(data, device):
    if isinstance(data, (list, tuple)):
        return [d.to(device, non_blocking=True) for d in data]
    return data.to(device, non_blocking=True)
```

### Profiling and Optimization

```python
# Profile code execution
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    pass
```

---

## Security Considerations

### Input Validation

```python
def validate_input_data(data: np.ndarray) -> bool:
    """Validate input data for security and correctness."""
    
    # Check data type
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be numpy array")
    
    # Check data shape
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    
    # Check for NaN/Inf values
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or infinite values")
    
    # Check data range
    if np.abs(data).max() > 1e6:
        raise ValueError("Input values are too large")
    
    return True

def sanitize_file_path(path: str) -> Path:
    """Sanitize file path to prevent directory traversal."""
    path = Path(path).resolve()
    
    # Ensure path is within allowed directory
    allowed_dir = Path("data/").resolve()
    if not str(path).startswith(str(allowed_dir)):
        raise ValueError("Path outside allowed directory")
    
    return path
```

### Authentication Integration

```python
from functools import wraps
from src.security import SecurityManager

def require_authentication(func):
    """Decorator to require authentication."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = kwargs.get('auth_token')
        if not token:
            raise PermissionError("Authentication required")
        
        security = SecurityManager()
        is_valid, payload = security.verify_token(token)
        if not is_valid:
            raise PermissionError("Invalid authentication token")
        
        kwargs['user_id'] = payload['user_id']
        return func(*args, **kwargs)
    
    return wrapper

@require_authentication
def sensitive_operation(data, auth_token=None, user_id=None):
    """Perform sensitive operation with authentication."""
    # Operation implementation
    pass
```

### Secure Configuration

```python
import os
from pathlib import Path

class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment and files."""
        # Load from environment variables
        self.config['secret_key'] = os.getenv('SECRET_KEY')
        self.config['database_url'] = os.getenv('DATABASE_URL')
        
        # Validate required settings
        if not self.config['secret_key']:
            raise ValueError("SECRET_KEY environment variable required")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
```

---

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY setup.py .

# Install package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "src.api.server"]
```

### Kubernetes Deployment

```yaml
# deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exoplanet-pipeline
  labels:
    app: exoplanet-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: exoplanet-pipeline
  template:
    metadata:
      labels:
        app: exoplanet-pipeline
    spec:
      containers:
      - name: exoplanet-pipeline
        image: exoplanet-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Monitoring and Logging

```python
# src/utils/monitoring.py
import logging
import time
from functools import wraps
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor application performance."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def time_function(self, func_name: str = None):
        """Decorator to time function execution."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self._record_success(name, time.time() - start_time)
                    return result
                except Exception as e:
                    self._record_error(name, time.time() - start_time, str(e))
                    raise
            
            return wrapper
        return decorator
    
    def _record_success(self, name: str, duration: float):
        """Record successful function execution."""
        if name not in self.metrics:
            self.metrics[name] = {'calls': 0, 'total_time': 0, 'errors': 0}
        
        self.metrics[name]['calls'] += 1
        self.metrics[name]['total_time'] += duration
        
        self.logger.info(f"Function {name} completed in {duration:.3f}s")
    
    def _record_error(self, name: str, duration: float, error: str):
        """Record function execution error."""
        if name not in self.metrics:
            self.metrics[name] = {'calls': 0, 'total_time': 0, 'errors': 0}
        
        self.metrics[name]['errors'] += 1
        self.logger.error(f"Function {name} failed after {duration:.3f}s: {error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()
```

---

## Contributing

### Getting Started

1. **Fork the Repository**: Create your own fork on GitHub
2. **Clone Locally**: Clone your fork to your development machine
3. **Create Branch**: Create a feature branch for your changes
4. **Make Changes**: Implement your feature or fix
5. **Test**: Ensure all tests pass and add new tests if needed
6. **Document**: Update documentation as needed
7. **Submit PR**: Create a pull request with clear description

### Contribution Guidelines

#### Code Quality

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Maintain test coverage above 90%
- Use meaningful variable and function names

#### Testing Requirements

- All new features must include tests
- Bug fixes must include regression tests
- Tests should be fast and reliable
- Use appropriate test fixtures and mocks

#### Documentation Requirements

- Update API documentation for new features
- Add examples for complex functionality
- Update user guide if user-facing changes
- Include docstrings for all public functions

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer approval
3. **Testing**: Manual testing of new features
4. **Documentation**: Review of documentation updates
5. **Final Approval**: Maintainer final approval and merge

### Release Process

1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Run full test suite
4. **Documentation**: Update documentation
5. **Tag Release**: Create git tag
6. **Deploy**: Deploy to production environments
7. **Announce**: Announce release to community

---

## Additional Resources

### Useful Tools

- **Code Quality**: black, flake8, mypy, pre-commit
- **Testing**: pytest, pytest-cov, pytest-mock
- **Documentation**: sphinx, sphinx-rtd-theme
- **Profiling**: cProfile, memory_profiler, py-spy
- **Monitoring**: tensorboard, wandb, mlflow

### Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Clean Code Principles](https://clean-code-developer.com/)
- [Git Best Practices](https://git-scm.com/book)

### Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issue Tracker**: Report bugs and request features
- **Wiki**: Community-maintained documentation
- **Slack/Discord**: Real-time community chat

---

This developer guide is a living document. Please contribute improvements and keep it up to date as the project evolves.