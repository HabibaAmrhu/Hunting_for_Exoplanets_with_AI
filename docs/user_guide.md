# User Guide

## Exoplanet Detection Pipeline - Complete User Guide

This comprehensive guide will help you get started with the exoplanet detection pipeline, from installation to advanced usage.

### Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Deployment](#deployment)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Quick Start

Get up and running in 5 minutes:

```bash
# Clone the repository
git clone https://github.com/your-org/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Install dependencies
pip install -r requirements.txt

# Download sample data
python scripts/download_sample_data.py

# Train a baseline model
python scripts/train_baseline.py

# Run the web interface
streamlit run streamlit_app/main.py
```

---

## Installation

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for data and models
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Installation Methods

#### Method 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv exoplanet_env

# Activate environment (Windows)
exoplanet_env\Scripts\activate

# Activate environment (macOS/Linux)
source exoplanet_env/bin/activate

# Install package
pip install -r requirements.txt
```

#### Method 2: Docker

```bash
# Build Docker image
docker build -t exoplanet-pipeline .

# Run container
docker run -p 8501:8501 exoplanet-pipeline
```

#### Method 3: Development Installation

```bash
# Clone repository
git clone https://github.com/your-org/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Verify Installation

```python
import src
from src.models import ExoplanetCNN
from src.data import ExoplanetDataset

print("Installation successful!")
```

---

## Basic Usage

### 1. Data Loading

```python
from src.data import ExoplanetDataset
import torch

# Load dataset
dataset = ExoplanetDataset(
    data_dir="data/kepler_data",
    transform=None
)

# Create data loader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

print(f"Dataset size: {len(dataset)}")
```

### 2. Model Creation

```python
from src.models import ExoplanetCNN

# Create CNN model
model = ExoplanetCNN(
    input_length=1000,
    num_filters=64,
    dropout_rate=0.3
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

### 3. Training

```python
from src.training import ExoplanetTrainer
import torch.nn as nn

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Create trainer
trainer = ExoplanetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

# Train model
history = trainer.train(epochs=50, patience=10)
```

### 4. Inference

```python
from src.inference import ModelInference

# Load trained model
inference = ModelInference("models/best_model.pth")

# Make predictions
predictions = inference.predict(test_data)
print(f"Found {sum(predictions > 0.5)} potential exoplanets")
```

---

## Data Preparation

### Supported Data Formats

The pipeline supports multiple data formats:

1. **Kepler/K2 Light Curves**: FITS files from MAST
2. **TESS Light Curves**: FITS files from MAST
3. **Custom CSV**: Time, flux, flux_error columns
4. **Synthetic Data**: Generated using built-in tools

### Data Directory Structure

```
data/
├── kepler/
│   ├── raw/
│   │   ├── kplr001234567-2009131105131_llc.fits
│   │   └── ...
│   └── processed/
│       ├── light_curves.h5
│       └── metadata.csv
├── tess/
│   └── ...
└── synthetic/
    └── ...
```

### Data Download

#### Automatic Download

```python
from src.data import TESSDownloader

# Download TESS data
downloader = TESSDownloader()
downloader.download_sector_data(
    sector=1,
    output_dir="data/tess/sector_1"
)
```

#### Manual Download

1. Visit [MAST Portal](https://mast.stsci.edu/)
2. Search for light curve data
3. Download FITS files
4. Place in appropriate directory

### Data Preprocessing

```python
from src.preprocessing import LightCurvePreprocessor

# Create preprocessor
preprocessor = LightCurvePreprocessor(
    normalize=True,
    remove_outliers=True,
    fill_gaps=True
)

# Process light curve
processed_data = preprocessor.process(raw_light_curve)
```

### Data Augmentation

```python
from src.data import DataAugmentation

# Setup augmentation
augmenter = DataAugmentation(
    noise_level=0.01,
    time_shift_range=0.1,
    amplitude_scale_range=(0.9, 1.1)
)

# Apply augmentation
augmented_data = augmenter.augment(light_curve_data)
```

---

## Model Training

### Available Models

1. **CNN (Convolutional Neural Network)**: Best for general use
2. **Transformer**: Good for long sequences
3. **LSTM**: Good for temporal patterns
4. **Ensemble**: Combines multiple models
5. **Bayesian Models**: Provides uncertainty estimates

### Training Configuration

Create a configuration file `config.json`:

```json
{
  "model": {
    "type": "cnn",
    "input_length": 1000,
    "num_filters": 64,
    "dropout_rate": 0.3
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "patience": 15
  },
  "data": {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "augmentation": true
  }
}
```

### Training Scripts

#### Basic Training

```bash
python scripts/train_baseline.py --config config.json --output models/
```

#### Advanced Training

```bash
python scripts/train_advanced_models.py \
    --model transformer \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0001
```

### Hyperparameter Optimization

```python
from src.optimization import HyperparameterTuner, HyperparameterSpace

# Define search space
search_space = HyperparameterSpace(
    learning_rate=(1e-5, 1e-2),
    batch_size=[16, 32, 64, 128],
    dropout_rate=(0.1, 0.7)
)

# Create tuner
tuner = HyperparameterTuner(
    model_factory=lambda **kwargs: ExoplanetCNN(**kwargs),
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)

# Run optimization
results = tuner.tune(
    method='bayesian',
    search_space=search_space,
    n_trials=50
)
```

### Monitoring Training

#### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log metrics during training
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('F1/Validation', val_f1, epoch)
```

#### Real-time Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

---

## Evaluation and Analysis

### Model Evaluation

```python
from src.training import MetricsCalculator

# Calculate metrics
calculator = MetricsCalculator()
metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)

print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
```

### Model Comparison

```python
from src.evaluation import ModelComparator

# Compare multiple models
comparator = ModelComparator()
results = comparator.compare_models(
    models={'CNN': cnn_model, 'Transformer': transformer_model},
    test_loader=test_loader
)

# Generate comparison report
comparator.generate_report(results, "model_comparison.html")
```

### Explainability Analysis

```python
from src.explainability import IntegratedGradientsExplainer

# Create explainer
explainer = IntegratedGradientsExplainer(model, device)

# Generate explanations
attributions = explainer.explain(sample_data)

# Visualize explanations
explainer.visualize_attributions(sample_data, attributions)
```

### Performance Analysis

```python
from src.visualization import AdvancedVisualizer

# Create visualizer
visualizer = AdvancedVisualizer()

# Plot performance comparison
fig = visualizer.plot_performance_comparison(results)
fig.show()

# Plot learning curves
fig = visualizer.plot_learning_curves(training_history)
fig.show()
```

---

## Deployment

### Local Deployment

#### Streamlit Web App

```bash
# Run web interface
streamlit run streamlit_app/main.py

# Access at http://localhost:8501
```

#### REST API

```bash
# Start API server
python src/api/server.py

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 0.99, 1.01, ...]}'
```

### Production Deployment

#### Docker Deployment

```bash
# Build production image
docker build -f deployment/docker/Dockerfile.prod -t exoplanet-prod .

# Run with docker-compose
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

#### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -l app=exoplanet-pipeline
```

#### Cloud Deployment

##### AWS

```bash
# Deploy to AWS ECS
aws ecs create-service --cli-input-json file://deployment/aws/ecs-service.json

# Deploy to AWS Lambda
sam deploy --template-file deployment/aws/template.yaml
```

##### Google Cloud

```bash
# Deploy to Google Cloud Run
gcloud run deploy exoplanet-pipeline \
  --image gcr.io/project-id/exoplanet-pipeline \
  --platform managed
```

### Monitoring and Logging

```python
from src.monitoring import ModelMonitor

# Setup monitoring
monitor = ModelMonitor(
    model=model,
    reference_data=reference_data,
    alert_thresholds={'accuracy': 0.8}
)

# Monitor predictions
report = monitor.check_performance(predictions, targets)
if report['alerts']:
    print("Performance degradation detected!")
```

---

## Advanced Features

### Multi-Survey Data Processing

```python
from src.data import MultiSurveyPipeline

# Process data from multiple surveys
pipeline = MultiSurveyPipeline()
pipeline.add_survey('kepler', kepler_data)
pipeline.add_survey('tess', tess_data)

# Harmonize and combine data
combined_data = pipeline.process_all()
```

### Real-time Processing

```python
from src.realtime import StreamingProcessor

# Setup streaming processor
processor = StreamingProcessor(
    model=model,
    buffer_size=1000,
    processing_interval=60  # seconds
)

# Start processing stream
processor.start_stream(data_source)
```

### Bayesian Uncertainty Estimation

```python
from src.models import BayesianCNN

# Create Bayesian model
bayesian_model = BayesianCNN(
    input_length=1000,
    num_samples=100
)

# Get predictions with uncertainty
predictions, uncertainties = bayesian_model.predict_with_uncertainty(data)
```

### Custom Model Development

```python
import torch.nn as nn
from src.models.base import BaseExoplanetModel

class CustomModel(BaseExoplanetModel):
    def __init__(self, input_length):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: GPU memory exhausted during training.

**Solutions**:
```python
# Reduce batch size
batch_size = 16  # Instead of 32 or 64

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Slow Training

**Problem**: Training takes too long.

**Solutions**:
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. Poor Model Performance

**Problem**: Model accuracy is low.

**Solutions**:
- Increase model complexity
- Add data augmentation
- Tune hyperparameters
- Check data quality
- Use ensemble methods

#### 4. Data Loading Issues

**Problem**: Slow data loading or memory issues.

**Solutions**:
```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Increase workers
    pin_memory=True,  # For GPU training
    persistent_workers=True
)
```

### Performance Optimization

#### Memory Optimization

```python
# Clear cache regularly
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self.layer, x)
```

#### Speed Optimization

```python
# Compile model (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model)

# Use optimized data types
model = model.half()  # Use FP16
```

### Debugging

#### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

#### Profile Code

```python
import cProfile

def profile_training():
    # Your training code here
    pass

cProfile.run('profile_training()', 'training_profile.prof')
```

---

## Best Practices

### Data Management

1. **Version Control**: Use DVC for data versioning
2. **Validation**: Always validate data quality
3. **Backup**: Keep backups of important datasets
4. **Documentation**: Document data sources and preprocessing steps

### Model Development

1. **Baseline First**: Start with simple models
2. **Incremental Improvement**: Add complexity gradually
3. **Cross-Validation**: Use proper validation techniques
4. **Reproducibility**: Set random seeds and save configurations

### Code Quality

1. **Testing**: Write unit tests for critical functions
2. **Documentation**: Document all public APIs
3. **Type Hints**: Use type annotations
4. **Code Review**: Review code before merging

### Security

1. **Authentication**: Always use authentication in production
2. **Input Validation**: Validate all user inputs
3. **Secrets Management**: Never hardcode secrets
4. **Regular Updates**: Keep dependencies updated

### Performance

1. **Profiling**: Profile code to identify bottlenecks
2. **Caching**: Cache expensive computations
3. **Batch Processing**: Process data in batches
4. **Resource Monitoring**: Monitor CPU, memory, and GPU usage

---

## Getting Help

### Documentation

- [API Reference](api_reference.md)
- [Installation Guide](installation.md)
- [Examples](../notebooks/)

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Community-maintained documentation

### Support

For technical support, please:

1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include system information and error logs

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/exoplanet-detection-pipeline.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

---

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.