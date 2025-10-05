"""
Baseline 1D CNN model for exoplanet detection from light curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class ExoplanetCNN(nn.Module):
    """
    Baseline 1D CNN for exoplanet detection from light curve data.
    
    Architecture:
    - Input: 1D array length 2048 (optionally 2 channels: raw + phase-folded)
    - Conv1D(32, kernel=5) → ReLU → MaxPool(2)
    - Conv1D(64, kernel=5) → ReLU → MaxPool(2)  
    - Conv1D(128, kernel=3) → ReLU → MaxPool(2)
    - Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(1) → Sigmoid
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (1=raw only, 2=raw+phase-folded)
            sequence_length: Length of input sequences
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(ExoplanetCNN, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_conv_output_size(self):
        """Calculate the output size after convolutional layers."""
        # Simulate forward pass to get output size
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.sequence_length)
            x = self.pool(F.relu(self.conv1(x)))  # /2
            x = self.pool(F.relu(self.conv2(x)))  # /4
            x = self.pool(F.relu(self.conv3(x)))  # /8
            self.conv_output_size = x.numel()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Convolutional layers with ReLU and pooling
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor from FC1 layer
        """
        # Forward pass up to FC1
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and get features
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        
        return features
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model architecture information."""
        return {
            'model_type': 'ExoplanetCNN',
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': self.count_parameters(),
            'conv_output_size': self.conv_output_size
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in exoplanet detection.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (planets)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits or probabilities)
            targets: True labels (0 or 1)
            
        Returns:
            Focal loss value
        """
        # Ensure inputs are probabilities
        if inputs.max() > 1.0 or inputs.min() < 0.0:
            inputs = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        
        # Compute p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Compute focal weight
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = focal_weight * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Initialize weighted BCE loss.
        
        Args:
            pos_weight: Weight for positive class (planets)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Model predictions
            targets: True labels
            
        Returns:
            Weighted BCE loss
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs.squeeze(), targets.float(), pos_weight=pos_weight
            )
        else:
            return F.binary_cross_entropy(inputs.squeeze(), targets.float())


def create_cnn_model(
    input_channels: int = 2,
    sequence_length: int = 2048,
    dropout_rate: float = 0.5,
    use_batch_norm: bool = True
) -> ExoplanetCNN:
    """
    Factory function to create CNN model with standard configuration.
    
    Args:
        input_channels: Number of input channels
        sequence_length: Length of input sequences
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Configured ExoplanetCNN model
    """
    model = ExoplanetCNN(
        input_channels=input_channels,
        sequence_length=sequence_length,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    return model


def get_loss_function(
    loss_type: str = 'bce',
    class_weights: Optional[Tuple[float, float]] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Get appropriate loss function for exoplanet detection.
    
    Args:
        loss_type: Type of loss ('bce', 'weighted_bce', 'focal')
        class_weights: Weights for (negative, positive) classes
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        
    Returns:
        Loss function module
    """
    if loss_type == 'bce':
        return nn.BCELoss()
    
    elif loss_type == 'weighted_bce':
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
        else:
            pos_weight = None
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def model_summary(model: ExoplanetCNN, input_shape: Tuple[int, int, int]) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: The CNN model
        input_shape: Shape of input tensor (batch_size, channels, sequence_length)
        
    Returns:
        String summary of the model
    """
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("ExoplanetCNN Model Summary")
    summary_lines.append("=" * 60)
    
    # Model info
    info = model.get_model_info()
    summary_lines.append(f"Model Type: {info['model_type']}")
    summary_lines.append(f"Input Channels: {info['input_channels']}")
    summary_lines.append(f"Sequence Length: {info['sequence_length']}")
    summary_lines.append(f"Dropout Rate: {info['dropout_rate']}")
    summary_lines.append(f"Batch Normalization: {info['use_batch_norm']}")
    summary_lines.append(f"Total Parameters: {info['total_parameters']:,}")
    
    summary_lines.append("-" * 60)
    summary_lines.append("Layer Architecture:")
    summary_lines.append("-" * 60)
    
    # Simulate forward pass to get layer shapes
    model.eval()
    with torch.no_grad():
        x = torch.zeros(input_shape)
        
        # Conv layers
        x = model.conv1(x)
        summary_lines.append(f"Conv1D(32, k=5): {list(x.shape)}")
        x = model.pool(F.relu(x))
        summary_lines.append(f"MaxPool1D(2): {list(x.shape)}")
        
        x = model.conv2(x)
        summary_lines.append(f"Conv1D(64, k=5): {list(x.shape)}")
        x = model.pool(F.relu(x))
        summary_lines.append(f"MaxPool1D(2): {list(x.shape)}")
        
        x = model.conv3(x)
        summary_lines.append(f"Conv1D(128, k=3): {list(x.shape)}")
        x = model.pool(F.relu(x))
        summary_lines.append(f"MaxPool1D(2): {list(x.shape)}")
        
        # Flatten
        x = x.view(x.size(0), -1)
        summary_lines.append(f"Flatten: {list(x.shape)}")
        
        # FC layers
        x = model.fc1(x)
        summary_lines.append(f"Linear(128): {list(x.shape)}")
        
        x = model.fc2(x)
        summary_lines.append(f"Linear(1): {list(x.shape)}")
    
    summary_lines.append("=" * 60)
    
    return "\n".join(summary_lines)