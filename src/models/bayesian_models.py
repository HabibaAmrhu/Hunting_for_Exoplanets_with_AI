"""
Bayesian neural networks for uncertainty quantification in exoplanet detection.
Implements variational inference and Monte Carlo dropout techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with variational inference.
    
    Implements weight uncertainty using variational Bayes approach.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        bias: bool = True
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_std: Standard deviation of weight prior
            bias: Whether to include bias term
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_logvar = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logvar', None)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        # Initialize weight mean
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        
        # Initialize weight log variance
        nn.init.constant_(self.weight_logvar, -5.0)  # Small initial variance
        
        # Initialize bias
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_logvar, -5.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps
        
        # Sample bias
        bias = None
        if self.bias_mu is not None:
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        Returns:
            KL divergence
        """
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 + 
            weight_var / self.prior_std**2 - 
            self.weight_logvar + 
            math.log(self.prior_std**2) - 1
        )
        
        # KL divergence for bias
        bias_kl = 0.0
        if self.bias_mu is not None:
            bias_var = torch.exp(self.bias_logvar)
            bias_kl = 0.5 * torch.sum(
                self.bias_mu**2 / self.prior_std**2 + 
                bias_var / self.prior_std**2 - 
                self.bias_logvar + 
                math.log(self.prior_std**2) - 1
            )
        
        return weight_kl + bias_kl


class BayesianCNN(nn.Module):
    """
    Bayesian CNN for exoplanet detection with uncertainty quantification.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 2048,
        prior_std: float = 1.0,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Bayesian CNN.
        
        Args:
            input_channels: Number of input channels
            sequence_length: Length of input sequences
            prior_std: Prior standard deviation for Bayesian layers
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.prior_std = prior_std
        
        # Convolutional layers (deterministic)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Calculate feature size
        self.feature_size = self._calculate_feature_size()
        
        # Bayesian fully connected layers
        self.fc1 = BayesianLinear(self.feature_size, 128, prior_std)
        self.fc2 = BayesianLinear(128, 64, prior_std)
        self.fc3 = BayesianLinear(64, 1, prior_std)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def _calculate_feature_size(self) -> int:
        """Calculate feature size after convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.sequence_length)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Bayesian fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers.
        
        Returns:
            Total KL divergence
        """
        kl_div = 0.0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_div += module.kl_divergence()
        return kl_div
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, uncertainty estimates)
        """
        self.train()  # Enable dropout and weight sampling
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class MCDropoutModel(nn.Module):
    """
    Monte Carlo Dropout model for uncertainty quantification.
    
    Uses dropout at inference time to estimate model uncertainty.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout model.
        
        Args:
            base_model: Base neural network model
            dropout_rate: Dropout rate for uncertainty estimation
        """
        super().__init__()
        
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Add dropout layers to the model
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add dropout layers to the base model."""
        # This is a simplified implementation
        # In practice, you'd need to modify the base model architecture
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model."""
        return self.base_model(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with MC Dropout uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, uncertainty estimates)
        """
        # Enable dropout during inference
        self.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class UncertaintyCalibrator:
    """
    Calibrator for uncertainty estimates to improve reliability.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.calibration_params = None
    
    def fit(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ):
        """
        Fit calibration parameters.
        
        Args:
            predictions: Model predictions
            uncertainties: Model uncertainty estimates
            targets: True targets
        """
        # Simple temperature scaling for calibration
        # In practice, you'd implement more sophisticated methods
        
        # Calculate calibration error
        errors = np.abs(predictions - targets)
        
        # Fit relationship between uncertainty and error
        # This is a simplified implementation
        self.calibration_params = {
            'scale': np.mean(errors) / np.mean(uncertainties),
            'bias': np.mean(errors) - np.mean(uncertainties)
        }
    
    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Calibrate uncertainty estimates.
        
        Args:
            uncertainties: Raw uncertainty estimates
            
        Returns:
            Calibrated uncertainty estimates
        """
        if self.calibration_params is None:
            return uncertainties
        
        calibrated = (
            uncertainties * self.calibration_params['scale'] + 
            self.calibration_params['bias']
        )
        
        return np.maximum(calibrated, 0.0)  # Ensure non-negative


# Factory functions
def create_bayesian_cnn(
    input_channels: int = 2,
    sequence_length: int = 2048,
    **kwargs
) -> BayesianCNN:
    """Create Bayesian CNN with default configuration."""
    return BayesianCNN(
        input_channels=input_channels,
        sequence_length=sequence_length,
        **kwargs
    )


def create_mc_dropout_model(
    base_model: nn.Module,
    dropout_rate: float = 0.1
) -> MCDropoutModel:
    """Create MC Dropout model wrapper."""
    return MCDropoutModel(base_model, dropout_rate)


def create_uncertainty_calibrator() -> UncertaintyCalibrator:
    """Create uncertainty calibrator."""
    return UncertaintyCalibrator()