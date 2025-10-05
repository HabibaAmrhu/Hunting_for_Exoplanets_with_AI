# Installation Guide

This guide provides detailed installation instructions for the Physics-Informed Exoplanet Detection Pipeline across different environments and platforms.

## System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.8 or 3.9
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB free space for datasets and models

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# 2. Create virtual environment (recommended)
python -m venv exoplanet-env
source exoplanet-env/bin/activate  # On Windows: exoplanet-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python run_quick_test.py
```

### Method 2: Development Installation

```bash
# 1. Clone with development dependencies
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# 2. Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# 3. Run full test suite
python run_tests.py all --coverage
```

### Method 3: Conda Installation

```bash
# 1. Create conda environment
conda create -n exoplanet python=3.8
conda activate exoplanet

# 2. Install PyTorch with conda
conda install pytorch torchvision torchaudio -c pytorch

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Platform-Specific Instructions

### Windows

#### Prerequisites
```powershell
# Install Python from python.org or Microsoft Store
# Ensure pip is up to date
python -m pip install --upgrade pip

# Install Git for Windows if not already installed
# Download from: https://git-scm.com/download/win
```

#### Installation
```powershell
# Clone repository
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python run_quick_test.py --verbose
```

#### Common Windows Issues
- **Long path names**: Enable long path support in Windows settings
- **Permission errors**: Run command prompt as administrator if needed
- **CUDA issues**: Install NVIDIA drivers and CUDA toolkit separately

### macOS

#### Prerequisites
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew
brew install python@3.8

# Install Git
brew install git
```

#### Installation
```bash
# Clone repository
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python run_quick_test.py --verbose
```

#### macOS-Specific Notes
- **M1/M2 Macs**: Use conda-forge for better ARM64 support
- **Xcode**: Install Xcode command line tools: `xcode-select --install`

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv git

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev python3-dev
```

#### Installation
```bash
# Clone repository
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Test installation
python run_quick_test.py --verbose
```

## Cloud Platform Installation

### Google Colab

```python
# Install in Colab notebook
!git clone https://github.com/your-username/exoplanet-detection-pipeline.git
%cd exoplanet-detection-pipeline

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')

# Quick test
!python run_quick_test.py --verbose
```

### Kaggle Notebooks

```python
# Clone repository in Kaggle
import os
os.system('git clone https://github.com/your-username/exoplanet-detection-pipeline.git')
os.chdir('exoplanet-detection-pipeline')

# Install dependencies
os.system('pip install -r requirements.txt')

# Enable internet for data download
# (Ensure internet is enabled in Kaggle notebook settings)
```

### AWS/Azure/GCP

```bash
# Create VM instance with Python 3.8+
# SSH into instance

# Install dependencies
sudo apt update
sudo apt install python3-pip git

# Clone and install
git clone https://github.com/your-username/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline
pip3 install -r requirements.txt

# For GPU instances, install CUDA
# Follow NVIDIA CUDA installation guide for your platform
```

## GPU Setup (Optional)

### NVIDIA CUDA Installation

#### Windows
1. Download CUDA Toolkit from NVIDIA website
2. Install NVIDIA drivers (latest version)
3. Install CUDA Toolkit 11.8 or 12.x
4. Verify installation: `nvidia-smi`

#### Linux
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-470

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Verify installation
nvidia-smi
nvcc --version
```

#### PyTorch GPU Support
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dependency Details

### Core Dependencies
```
torch>=1.12.0          # Deep learning framework
torchvision>=0.13.0    # Computer vision utilities
scikit-learn>=1.0.0    # Machine learning tools
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
tqdm>=4.62.0           # Progress bars
```

### Optional Dependencies
```
plotly>=5.0.0          # Interactive plotting
streamlit>=1.15.0      # Web application framework
pytest>=6.0.0          # Testing framework
pytest-cov>=3.0.0     # Coverage reporting
jupyter>=1.0.0         # Notebook environment
psutil>=5.8.0          # System monitoring
```

### Development Dependencies
```
black>=22.0.0          # Code formatting
flake8>=4.0.0          # Code linting
mypy>=0.950            # Type checking
pre-commit>=2.15.0     # Git hooks
sphinx>=4.0.0          # Documentation generation
```

## Verification and Testing

### Quick Verification
```bash
# Basic functionality test (30 seconds)
python run_quick_test.py

# Verbose output with system info
python run_quick_test.py --verbose

# Include GPU testing
python run_quick_test.py --gpu --verbose
```

### Comprehensive Testing
```bash
# Run all tests (10-15 minutes)
python run_tests.py all

# Run with coverage report
python run_tests.py all --coverage --html

# Quick test suite only
python run_tests.py unit --quick
```

### Manual Verification
```python
# Test core imports
import torch
import numpy as np
import pandas as pd
from src.models.cnn import ExoplanetCNN
from src.data.dataset import LightCurveDataset

# Test model creation
model = ExoplanetCNN(input_channels=2, sequence_length=2048)
print(f"Model created with {model.count_parameters():,} parameters")

# Test GPU availability (if applicable)
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
else:
    print("Using CPU mode")
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to your script:
import sys
sys.path.insert(0, 'src')
```

#### Memory Issues
```bash
# Issue: Out of memory during training
# Solution: Reduce batch size or use CPU
python scripts/train_baseline.py --batch-size 16 --device cpu
```

#### CUDA Issues
```bash
# Issue: CUDA out of memory
# Solution: Clear cache and reduce batch size
python -c "import torch; torch.cuda.empty_cache()"

# Issue: CUDA version mismatch
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Permission Issues (Windows)
```powershell
# Run as administrator or adjust permissions
# Enable long path names in Windows settings
```

### Getting Help

1. **Check system requirements** - Ensure your system meets minimum requirements
2. **Update dependencies** - Run `pip install --upgrade -r requirements.txt`
3. **Clear cache** - Delete `__pycache__` directories and reinstall
4. **Check logs** - Look for detailed error messages in console output
5. **Create issue** - Report bugs on GitHub with system info and error logs

### Performance Optimization

#### CPU Optimization
```bash
# Set number of threads for CPU training
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### Memory Optimization
```python
# Reduce memory usage in training
trainer = ExoplanetTrainer(
    batch_size=16,  # Reduce from default 32
    gradient_accumulation_steps=2,  # Simulate larger batches
    mixed_precision=True  # Use automatic mixed precision
)
```

## Next Steps

After successful installation:

1. **Run Quick Tutorial**: `jupyter notebook notebooks/01_train_baseline.ipynb`
2. **Explore Web App**: `streamlit run streamlit_app/main.py`
3. **Train Your First Model**: `python scripts/train_baseline.py`
4. **Read Documentation**: Browse the `docs/` directory for detailed guides

## Support

If you encounter issues during installation:

- 📖 Check the [FAQ](docs/faq.md)
- 🐛 Report bugs via [GitHub Issues](https://github.com/your-username/exoplanet-detection-pipeline/issues)
- 💬 Ask questions in [GitHub Discussions](https://github.com/your-username/exoplanet-detection-pipeline/discussions)
- 📧 Contact maintainers for urgent issues
