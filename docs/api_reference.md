# API Reference

## Exoplanet Detection Pipeline API

This document provides comprehensive API reference for the exoplanet detection pipeline.

### Table of Contents

1. [Authentication](#authentication)
2. [Data Processing](#data-processing)
3. [Model Training](#model-training)
4. [Inference](#inference)
5. [Monitoring](#monitoring)
6. [Utilities](#utilities)

---

## Authentication

### SecurityManager

Main class for handling authentication and authorization.

```python
from src.security import SecurityManager, UserRole, Permission

# Initialize security manager
security = SecurityManager(
    secret_key="your-secret-key",
    token_expiry_hours=24,
    max_failed_attempts=5,
    lockout_duration_minutes=30
)
```

#### Methods

##### `create_user(username, email, password, role, admin_user_id=None)`

Create a new user account.

**Parameters:**
- `username` (str): Unique username
- `email` (str): User email address
- `password` (str): Plain text password (will be hashed)
- `role` (UserRole): User role (ADMIN, RESEARCHER, ANALYST, VIEWER)
- `admin_user_id` (str, optional): ID of admin creating the user

**Returns:**
- `Tuple[bool, str]`: Success status and message

**Example:**
```python
success, message = security.create_user(
    "researcher1",
    "researcher@example.com",
    "SecurePass123!",
    UserRole.RESEARCHER
)
```

##### `authenticate_user(username, password, ip_address="127.0.0.1")`

Authenticate user credentials.

**Parameters:**
- `username` (str): Username or email
- `password` (str): Plain text password
- `ip_address` (str): Client IP address

**Returns:**
- `Tuple[bool, Optional[str], Optional[str]]`: Success, user_id, error_message

##### `generate_token(user_id)`

Generate JWT token for authenticated user.

**Parameters:**
- `user_id` (str): User ID

**Returns:**
- `str`: JWT token

##### `verify_token(token)`

Verify JWT token validity.

**Parameters:**
- `token` (str): JWT token

**Returns:**
- `Tuple[bool, Optional[Dict]]`: Validity status and payload

---

## Data Processing

### ExoplanetDataset

Main dataset class for handling exoplanet light curve data.

```python
from src.data import ExoplanetDataset

dataset = ExoplanetDataset(
    data_dir="path/to/data",
    transform=None,
    target_transform=None
)
```

#### Methods

##### `__init__(data_dir, transform=None, target_transform=None)`

Initialize dataset.

**Parameters:**
- `data_dir` (str): Path to data directory
- `transform` (callable, optional): Transform function for data
- `target_transform` (callable, optional): Transform function for targets

##### `__len__()`

Get dataset size.

**Returns:**
- `int`: Number of samples

##### `__getitem__(idx)`

Get sample by index.

**Parameters:**
- `idx` (int): Sample index

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: Data and target tensors

### DataAugmentation

Data augmentation utilities for light curve data.

```python
from src.data import DataAugmentation

augmenter = DataAugmentation(
    noise_level=0.01,
    time_shift_range=0.1,
    amplitude_scale_range=(0.9, 1.1)
)
```

#### Methods

##### `add_noise(data, noise_level=None)`

Add Gaussian noise to light curve.

**Parameters:**
- `data` (np.ndarray): Light curve data
- `noise_level` (float, optional): Noise standard deviation

**Returns:**
- `np.ndarray`: Augmented data

##### `time_shift(data, shift_range=None)`

Apply random time shift to light curve.

**Parameters:**
- `data` (np.ndarray): Light curve data
- `shift_range` (float, optional): Maximum shift fraction

**Returns:**
- `np.ndarray`: Shifted data

---

## Model Training

### ExoplanetTrainer

Main training class for exoplanet detection models.

```python
from src.training import ExoplanetTrainer

trainer = ExoplanetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)
```

#### Methods

##### `train(epochs, patience=10, verbose=True)`

Train the model.

**Parameters:**
- `epochs` (int): Number of training epochs
- `patience` (int): Early stopping patience
- `verbose` (bool): Print training progress

**Returns:**
- `Dict`: Training history with losses and metrics

##### `evaluate(data_loader)`

Evaluate model on dataset.

**Parameters:**
- `data_loader` (DataLoader): Data loader for evaluation

**Returns:**
- `Dict`: Evaluation metrics

##### `save_checkpoint(filepath, epoch, metrics)`

Save training checkpoint.

**Parameters:**
- `filepath` (str): Path to save checkpoint
- `epoch` (int): Current epoch
- `metrics` (dict): Current metrics

### MetricsCalculator

Calculate evaluation metrics for binary classification.

```python
from src.training import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
```

#### Methods

##### `calculate_metrics(y_true, y_pred, y_prob=None)`

Calculate comprehensive metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `y_prob` (np.ndarray, optional): Prediction probabilities

**Returns:**
- `Dict`: Dictionary with all metrics

---

## Models

### ExoplanetCNN

Convolutional Neural Network for exoplanet detection.

```python
from src.models import ExoplanetCNN

model = ExoplanetCNN(
    input_length=1000,
    num_filters=32,
    filter_sizes=[3, 5, 7],
    dropout_rate=0.3,
    use_batch_norm=True
)
```

#### Parameters

- `input_length` (int): Length of input time series
- `num_filters` (int): Number of convolutional filters
- `filter_sizes` (List[int]): Sizes of convolutional filters
- `dropout_rate` (float): Dropout probability
- `use_batch_norm` (bool): Whether to use batch normalization

### TransformerModel

Transformer-based model for sequence modeling.

```python
from src.models import TransformerModel

model = TransformerModel(
    input_dim=1,
    d_model=128,
    nhead=8,
    num_layers=6,
    dropout=0.1
)
```

### EnsembleModel

Ensemble of multiple models for improved performance.

```python
from src.models import EnsembleModel

ensemble = EnsembleModel(
    models=[model1, model2, model3],
    weights=[0.4, 0.3, 0.3],
    voting_strategy='soft'
)
```

---

## Inference

### ModelInference

High-level inference interface.

```python
from src.inference import ModelInference

inference = ModelInference(
    model_path="path/to/model.pth",
    device="cuda"
)
```

#### Methods

##### `predict(data)`

Make predictions on data.

**Parameters:**
- `data` (np.ndarray or torch.Tensor): Input data

**Returns:**
- `np.ndarray`: Predictions

##### `predict_with_uncertainty(data)`

Make predictions with uncertainty estimates.

**Parameters:**
- `data` (np.ndarray or torch.Tensor): Input data

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Predictions and uncertainties

---

## Monitoring

### ModelMonitor

Monitor model performance and data drift.

```python
from src.monitoring import ModelMonitor

monitor = ModelMonitor(
    model=model,
    reference_data=reference_data,
    alert_thresholds={'accuracy': 0.8, 'drift_score': 0.1}
)
```

#### Methods

##### `check_performance(predictions, targets)`

Check model performance against thresholds.

**Parameters:**
- `predictions` (np.ndarray): Model predictions
- `targets` (np.ndarray): True targets

**Returns:**
- `Dict`: Performance report

##### `detect_drift(new_data)`

Detect data drift in new samples.

**Parameters:**
- `new_data` (np.ndarray): New data samples

**Returns:**
- `Dict`: Drift detection report

---

## Utilities

### Performance Optimization

```python
from src.utils import PerformanceOptimizer

optimizer = PerformanceOptimizer(device="cuda")

# Optimize model for inference
optimized_model = optimizer.optimize_model_for_inference(model)

# Create optimized data loader
dataloader = optimizer.create_optimized_dataloader(
    dataset, batch_size=32
)
```

### Visualization

```python
from src.visualization import AdvancedVisualizer

visualizer = AdvancedVisualizer()

# Create performance comparison plot
fig = visualizer.plot_performance_comparison(results)

# Create learning curves
fig = visualizer.plot_learning_curves(training_history)
```

---

## Error Handling

All API functions use consistent error handling patterns:

### Common Exceptions

- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Missing required files
- `RuntimeError`: Runtime errors during execution
- `PermissionError`: Insufficient permissions for operation

### Example Error Handling

```python
try:
    success, user_id, error = security.authenticate_user(username, password)
    if not success:
        print(f"Authentication failed: {error}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Configuration

### Environment Variables

- `EXOPLANET_DATA_DIR`: Default data directory
- `EXOPLANET_MODEL_DIR`: Default model directory
- `EXOPLANET_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `EXOPLANET_DEVICE`: Default device (cpu, cuda)

### Configuration Files

Configuration can be provided via JSON files:

```json
{
  "data": {
    "batch_size": 32,
    "num_workers": 4,
    "augmentation": {
      "noise_level": 0.01,
      "time_shift_range": 0.1
    }
  },
  "training": {
    "learning_rate": 0.001,
    "epochs": 100,
    "patience": 10
  },
  "model": {
    "architecture": "cnn",
    "dropout_rate": 0.3,
    "use_batch_norm": true
  }
}
```

---

## Examples

### Complete Training Pipeline

```python
import torch
from src.data import ExoplanetDataset, DataAugmentation
from src.models import ExoplanetCNN
from src.training import ExoplanetTrainer, create_optimizer, create_scheduler
from src.security import create_security_manager, UserRole

# Initialize security
security = create_security_manager()

# Create user
security.create_user(
    "researcher", "researcher@example.com", 
    "SecurePass123!", UserRole.RESEARCHER
)

# Authenticate
success, user_id, _ = security.authenticate_user("researcher", "SecurePass123!")
token = security.generate_token(user_id)

# Load data
dataset = ExoplanetDataset("data/")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = ExoplanetCNN(input_length=1000)

# Create optimizer and scheduler
optimizer = create_optimizer(model, 'adamw', learning_rate=0.001)
scheduler = create_scheduler(optimizer, 'cosine', T_max=100)

# Create trainer
trainer = ExoplanetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=train_loader,  # Use same for demo
    criterion=torch.nn.BCELoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Train model
history = trainer.train(epochs=10)
print(f"Training completed. Best F1: {max([m['f1_score'] for m in history['val_metrics']])}")
```

### Batch Inference

```python
from src.inference import ModelInference
from src.utils import BatchProcessor

# Load model
inference = ModelInference("models/best_model.pth")

# Create batch processor
processor = BatchProcessor(
    model=inference.model,
    device=inference.device,
    batch_size=64
)

# Process large dataset
predictions, uncertainties = processor.process_dataset(test_dataset)
print(f"Processed {len(predictions)} samples")
```

---

## Version Information

- **API Version**: 1.0.0
- **Python Version**: 3.8+
- **PyTorch Version**: 1.9+
- **Required Dependencies**: See `requirements.txt`

For more detailed examples and tutorials, see the `notebooks/` directory.