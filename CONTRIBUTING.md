# Contributing to Physics-Informed Exoplanet Detection Pipeline

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to our physics-informed exoplanet detection pipeline.

## 🌟 Ways to Contribute

### 🔬 Research Contributions
- **Novel Augmentation Techniques**: Develop new physics-informed data augmentation methods
- **Model Architectures**: Implement advanced neural network designs for time series analysis
- **Evaluation Metrics**: Create new performance assessment methods for astronomical applications
- **Scientific Validation**: Validate methods on new datasets or astronomical surveys

### 💻 Technical Contributions
- **Code Improvements**: Optimize performance, fix bugs, improve code quality
- **Testing**: Add test cases, improve coverage, enhance validation frameworks
- **Documentation**: Write tutorials, improve API docs, create examples
- **Infrastructure**: CI/CD improvements, deployment options, containerization

### 📚 Educational Contributions
- **Tutorials**: Create learning materials for different skill levels
- **Examples**: Develop real-world use cases and applications
- **Workshops**: Design educational content for conferences and courses
- **Outreach**: Help make exoplanet science accessible to broader audiences

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/HabibaAmrhu/-Exoplanet-Detection.git
   cd exoplanet-detection-pipeline
   
   # Add upstream remote
   git remote add upstream https://github.com/HabibaAmrhu/-Exoplanet-Detection.git
   ```

2. **Create Development Environment**
   ```bash
   # Create virtual environment
   python -m venv dev-env
   source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   pip install -e .  # Install in development mode
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   python run_tests.py all --quick
   
   # Run code quality checks
   pre-commit install
   pre-commit run --all-files
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run relevant tests
   python run_tests.py unit
   python run_tests.py integration --quick
   
   # Check code quality
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add physics-informed noise modeling"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Open a PR on GitHub with a clear description
   - Link any related issues
   - Ensure all CI checks pass

## 📋 Contribution Guidelines

### Code Style

We follow Python best practices and use automated tools for consistency:

#### Formatting
```bash
# Use Black for code formatting
black src/ tests/ scripts/

# Use isort for import sorting
isort src/ tests/ scripts/
```

#### Linting
```bash
# Use flake8 for linting
flake8 src/ tests/ scripts/

# Use mypy for type checking
mypy src/
```

#### Example Code Style
```python
"""Module docstring describing the purpose."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from src.models.base import BaseModel


class ExampleModel(BaseModel):
    """
    Example model class following our style guidelines.
    
    Args:
        input_dim: Input dimension size
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Initialize layers
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the model architecture."""
        self.linear = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        x = self.linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x
```

### Testing Requirements

All contributions must include appropriate tests:

#### Unit Tests
```python
import pytest
import torch
from src.models.example import ExampleModel


class TestExampleModel:
    """Test suite for ExampleModel."""
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        model = ExampleModel(input_dim=10, hidden_dim=20)
        assert model.input_dim == 10
        assert model.hidden_dim == 20
    
    def test_forward_pass(self):
        """Test forward pass functionality."""
        model = ExampleModel(input_dim=10, hidden_dim=20)
        x = torch.randn(5, 10)
        output = model(x)
        
        assert output.shape == (5, 20)
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = ExampleModel(input_dim=10, hidden_dim=20)
        x = torch.randn(batch_size, 10)
        output = model(x)
        
        assert output.shape == (batch_size, 20)
```

#### Integration Tests
```python
def test_end_to_end_training():
    """Test complete training pipeline."""
    # Create mock data
    data, labels = create_mock_dataset(n_samples=100)
    
    # Initialize model and trainer
    model = ExampleModel(input_dim=10, hidden_dim=20)
    trainer = create_trainer(model, data, labels)
    
    # Train for one epoch
    history = trainer.train(epochs=1)
    
    # Verify training completed successfully
    assert len(history['train_loss']) == 1
    assert history['train_loss'][0] > 0
```

### Documentation Standards

#### Docstring Format
We use Google-style docstrings:

```python
def process_light_curve(
    time: np.ndarray, 
    flux: np.ndarray, 
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process a light curve with normalization and quality metrics.
    
    This function performs standard preprocessing on astronomical light curves,
    including optional normalization and quality assessment.
    
    Args:
        time: Time array in days (shape: [N])
        flux: Flux measurements (shape: [N])
        normalize: Whether to normalize flux to unit variance
        
    Returns:
        Tuple containing:
            - Processed flux array (shape: [N])
            - Quality metrics dictionary with keys:
                - 'rms': Root mean square of flux
                - 'mad': Median absolute deviation
                - 'completeness': Fraction of valid measurements
    
    Raises:
        ValueError: If time and flux arrays have different lengths
        
    Example:
        >>> time = np.linspace(0, 100, 1000)
        >>> flux = np.random.normal(1.0, 0.01, 1000)
        >>> processed_flux, metrics = process_light_curve(time, flux)
        >>> print(f"RMS: {metrics['rms']:.4f}")
    """
```

#### README Updates
When adding new features, update relevant documentation:

- Add to main README.md if it's a major feature
- Create specific documentation in docs/ directory
- Update API reference if adding new public functions
- Add examples to notebooks/ if appropriate

### Commit Message Guidelines

We follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `ci`: CI/CD changes

#### Examples
```
feat(models): add transformer architecture for time series

Implement 1D transformer model with positional encoding
optimized for astronomical light curve analysis.

Closes #123

fix(preprocessing): handle missing values in light curves

Add interpolation for gaps smaller than 5 data points
and masking for larger gaps to prevent artifacts.

docs(api): update synthetic transit generation examples

Add comprehensive examples showing parameter sampling
and injection workflow with real Kepler data.

test(integration): add cross-platform compatibility tests

Ensure pipeline works correctly on Windows, macOS, and Linux
with comprehensive environment validation.
```

## 🔬 Research Contribution Guidelines

### Scientific Rigor
- **Literature Review**: Cite relevant papers and compare with existing methods
- **Methodology**: Clearly describe new algorithms and their theoretical basis
- **Validation**: Provide comprehensive experimental validation
- **Reproducibility**: Include all code and parameters needed to reproduce results

### Data and Experiments
- **Datasets**: Use standard benchmarks when possible, document any new datasets
- **Baselines**: Compare against established methods in the field
- **Statistical Significance**: Report confidence intervals and statistical tests
- **Ablation Studies**: Demonstrate the contribution of each component

### Example Research Contribution Structure
```python
"""
Novel Physics-Informed Augmentation Method

This module implements a new augmentation technique based on 
stellar evolution models and observational constraints.

References:
    - Smith et al. (2023): "Stellar Variability in Exoplanet Surveys"
    - Jones & Brown (2022): "Physics-Based Data Augmentation"
"""

class StellarEvolutionAugmentation:
    """
    Augmentation based on stellar evolution models.
    
    This method generates realistic stellar variability patterns
    based on stellar mass, age, and metallicity using theoretical
    models from stellar evolution codes.
    
    The approach is validated against known variable stars in the
    Kepler catalog and shows improved generalization compared to
    traditional noise-based augmentation methods.
    
    Args:
        evolution_model: Stellar evolution model to use ('MESA', 'PARSEC')
        variability_amplitude: Maximum variability amplitude (default: 0.01)
        
    References:
        Paxton et al. (2011): "Modules for Experiments in Stellar Astrophysics"
    """
```

## 🧪 Testing New Contributions

### Before Submitting
1. **Run Full Test Suite**
   ```bash
   python run_tests.py all --coverage
   ```

2. **Performance Testing**
   ```bash
   python run_tests.py performance
   ```

3. **Cross-Platform Testing** (if possible)
   ```bash
   # Test on different Python versions
   python3.7 run_quick_test.py
   python3.8 run_quick_test.py
   python3.9 run_quick_test.py
   ```

4. **Memory and Speed Profiling**
   ```bash
   # Profile memory usage
   python -m memory_profiler scripts/train_baseline.py
   
   # Profile execution time
   python -m cProfile -o profile.stats scripts/train_baseline.py
   ```

### Continuous Integration
Our CI pipeline automatically runs:
- Unit and integration tests on multiple Python versions
- Code quality checks (linting, formatting, type checking)
- Performance benchmarks
- Cross-platform compatibility tests
- Documentation building and link checking

## 📚 Documentation Contributions

### Types of Documentation
1. **API Documentation**: Function and class docstrings
2. **Tutorials**: Step-by-step guides for specific tasks
3. **How-to Guides**: Solutions to common problems
4. **Reference**: Comprehensive technical documentation
5. **Examples**: Real-world use cases and applications

### Writing Guidelines
- **Clear and Concise**: Use simple language and short sentences
- **Practical Examples**: Include code examples that users can run
- **Visual Aids**: Add plots, diagrams, and screenshots when helpful
- **Cross-References**: Link to related documentation and external resources
- **Up-to-Date**: Ensure examples work with current code version

### Building Documentation Locally
```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View in browser
open _build/html/index.html
```

## 🎯 Priority Areas for Contribution

### High Priority
1. **Performance Optimization**: Speed up training and inference
2. **Memory Efficiency**: Reduce memory usage for large datasets
3. **New Datasets**: Support for TESS, K2, and ground-based surveys
4. **Advanced Models**: State-of-the-art architectures for time series

### Medium Priority
1. **Deployment Options**: Docker, cloud platforms, edge computing
2. **Visualization Tools**: Interactive plots and analysis dashboards
3. **Data Pipeline**: Streaming data processing and real-time analysis
4. **Educational Content**: Tutorials for different skill levels

### Research Opportunities
1. **Domain Adaptation**: Transfer learning between different surveys
2. **Active Learning**: Intelligent sample selection for labeling
3. **Uncertainty Quantification**: Improved confidence estimation
4. **Interpretability**: Advanced explainable AI techniques

## 🏆 Recognition

### Contributor Recognition
- Contributors are acknowledged in README.md and documentation
- Significant contributions may warrant co-authorship on papers
- Regular contributors may be invited to join the core team
- Outstanding contributions are highlighted in release notes

### Academic Credit
- Research contributions are properly cited and attributed
- Contributors are encouraged to publish their methods
- We support conference presentations and workshop submissions
- Collaboration opportunities with academic institutions

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and brainstorming
- **Email**: Direct contact for sensitive issues or collaboration
- **Slack/Discord**: Real-time chat (link in repository)

### Mentorship
New contributors can request mentorship for:
- Understanding the codebase architecture
- Learning astronomical data analysis techniques
- Developing research ideas and methodologies
- Navigating the contribution process

### Code Review Process
1. **Automated Checks**: CI pipeline runs automatically
2. **Peer Review**: At least one core team member reviews each PR
3. **Scientific Review**: Research contributions get additional scientific review
4. **Documentation Review**: Documentation changes are reviewed for clarity
5. **Final Approval**: Maintainers approve and merge contributions

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details on our community standards.

### Key Principles
- **Respectful Communication**: Treat all contributors with respect and kindness
- **Inclusive Environment**: Welcome contributors from all backgrounds and skill levels
- **Constructive Feedback**: Provide helpful, actionable feedback on contributions
- **Scientific Integrity**: Maintain high standards for research and methodology
- **Open Collaboration**: Share knowledge and help others learn and grow

## 🙏 Thank You

Thank you for contributing to advancing exoplanet science through innovative machine learning techniques! Your contributions help make this research accessible to the broader scientific community and advance our understanding of planetary systems beyond our own.

Every contribution, no matter how small, makes a difference. Whether you're fixing a typo, adding a test case, implementing a new feature, or conducting groundbreaking research, you're helping to push the boundaries of what's possible in exoplanet detection and characterization.

**Happy contributing! 🚀🌟**
