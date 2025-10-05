"""
Comprehensive unit tests for model architectures.
Tests model initialization, forward pass, gradient flow, and architecture consistency.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.cnn import ExoplanetCNN, FocalLoss, WeightedBCELoss, create_model, create_loss_function
from models.lstm import ExoplanetLSTM
from models.transformer import ExoplanetTransformer
from models.ensemble import EnsembleModel, UncertaintyQuantifier


class TestExoplanetCNN:
    """Test suite for ExoplanetCNN model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_channels = 2
        self.sequence_length = 2048
        self.batch_size = 4
        
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Default initialization
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        assert model.input_channels == self.input_channels
        assert model.sequence_length == self.sequence_length
        assert model.dropout_rate == 0.5
        assert model.use_batch_norm == True
        
        # Custom initialization
        model_custom = ExoplanetCNN(
            input_channels=1,
            sequence_length=1024,
            dropout_rate=0.3,
            use_batch_norm=False
        )
        
        assert model_custom.input_channels == 1
        assert model_custom.sequence_length == 1024
        assert model_custom.dropout_rate == 0.3
        assert model_custom.use_batch_norm == False
        
    def test_forward_pass(self):
        """Test forward pass with different input sizes."""
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        model.eval()
        
        # Test with batch
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        output = model(x)
        
        assert output.shape == (self.batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
        
        # Test with single sample
        x_single = torch.randn(1, self.input_channels, self.sequence_length)
        output_single = model(x_single)
        
        assert output_single.shape == (1, 1)
        
    def test_gradient_flow(self):
        """Test gradient computation and backpropagation."""
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length, requires_grad=True)
        target = torch.randint(0, 2, (self.batch_size,)).float()
        
        output = model(x).squeeze()
        loss = nn.BCELoss()(output, target)
        loss.backward()
        
        # Check that gradients exist and are non-zero
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        assert has_gradients, "Model should have non-zero gradients"
        
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        model.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        features = model.get_features(x)
        
        assert features.shape[0] == self.batch_size
        assert features.shape[1] == 128  # FC1 output size
        
    def test_parameter_count(self):
        """Test parameter counting functionality."""
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        param_count = model.count_parameters()
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert param_count == manual_count
        assert param_count > 0
        
    def test_model_info(self):
        """Test model information retrieval."""
        model = ExoplanetCNN(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        info = model.get_model_info()
        
        assert 'model_name' in info
        assert 'input_channels' in info
        assert 'sequence_length' in info
        assert 'total_parameters' in info
        assert info['input_channels'] == self.input_channels
        assert info['sequence_length'] == self.sequence_length


class TestLossFunction:
    """Test suite for custom loss functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.predictions = torch.sigmoid(torch.randn(self.batch_size))
        self.targets = torch.randint(0, 2, (self.batch_size,)).float()
        
    def test_focal_loss(self):
        """Test Focal Loss implementation."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        loss = focal_loss(self.predictions, self.targets)
        
        assert loss.item() >= 0
        assert loss.requires_grad
        
        # Test with different parameters
        focal_loss_custom = FocalLoss(alpha=0.5, gamma=1.0)
        loss_custom = focal_loss_custom(self.predictions, self.targets)
        
        assert loss_custom.item() >= 0
        
    def test_weighted_bce_loss(self):
        """Test Weighted BCE Loss implementation."""
        weighted_bce = WeightedBCELoss()
        
        loss = weighted_bce(self.predictions, self.targets)
        
        assert loss.item() >= 0
        assert loss.requires_grad
        
        # Test with custom pos_weight
        weighted_bce_custom = WeightedBCELoss(pos_weight=2.0)
        loss_custom = weighted_bce_custom(self.predictions, self.targets)
        
        assert loss_custom.item() >= 0
        
    def test_loss_factory(self):
        """Test loss function factory."""
        # Test BCE
        bce_loss = create_loss_function('bce')
        assert isinstance(bce_loss, nn.BCELoss)
        
        # Test Weighted BCE
        weighted_bce_loss = create_loss_function('weighted_bce')
        assert isinstance(weighted_bce_loss, WeightedBCELoss)
        
        # Test Focal Loss
        focal_loss = create_loss_function('focal', alpha=0.25, gamma=2.0)
        assert isinstance(focal_loss, FocalLoss)
        
        # Test invalid loss type
        with pytest.raises(ValueError):
            create_loss_function('invalid_loss')


class TestExoplanetLSTM:
    """Test suite for ExoplanetLSTM model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_channels = 2
        self.sequence_length = 2048
        self.batch_size = 4
        
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        model = ExoplanetLSTM(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        assert hasattr(model, 'cnn_features')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'classifier')
        
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        model = ExoplanetLSTM(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        model.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        output = model(x)
        
        assert output.shape == (self.batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    def test_lstm_gradient_flow(self):
        """Test LSTM gradient computation."""
        model = ExoplanetLSTM(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length, requires_grad=True)
        target = torch.randint(0, 2, (self.batch_size,)).float()
        
        output = model(x).squeeze()
        loss = nn.BCELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(param.grad is not None and torch.any(param.grad != 0) 
                          for param in model.parameters())
        assert has_gradients


class TestExoplanetTransformer:
    """Test suite for ExoplanetTransformer model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_channels = 2
        self.sequence_length = 2048
        self.batch_size = 4
        
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        model = ExoplanetTransformer(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'classifier')
        
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass."""
        model = ExoplanetTransformer(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        model.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        output = model(x)
        
        assert output.shape == (self.batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    def test_positional_encoding(self):
        """Test positional encoding functionality."""
        model = ExoplanetTransformer(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length
        )
        
        # Test that positional encoding has correct shape
        pos_encoding = model.positional_encoding
        expected_shape = (1, self.sequence_length, model.d_model)
        
        assert pos_encoding.shape == expected_shape


class TestEnsembleModel:
    """Test suite for EnsembleModel."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_channels = 2
        self.sequence_length = 2048
        self.batch_size = 4
        
        # Create individual models
        self.models = [
            ExoplanetCNN(self.input_channels, self.sequence_length),
            ExoplanetLSTM(self.input_channels, self.sequence_length),
            ExoplanetTransformer(self.input_channels, self.sequence_length)
        ]
        
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        ensemble = EnsembleModel(
            models=self.models,
            combination_method='average'
        )
        
        assert len(ensemble.models) == len(self.models)
        assert ensemble.combination_method == 'average'
        
    def test_ensemble_forward_pass(self):
        """Test ensemble forward pass."""
        ensemble = EnsembleModel(
            models=self.models,
            combination_method='average'
        )
        ensemble.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        output = ensemble(x)
        
        assert output.shape == (self.batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    def test_weighted_ensemble(self):
        """Test weighted ensemble combination."""
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleModel(
            models=self.models,
            combination_method='weighted',
            weights=weights
        )
        ensemble.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        output = ensemble(x)
        
        assert output.shape == (self.batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification."""
        ensemble = EnsembleModel(
            models=self.models,
            combination_method='uncertainty_weighted'
        )
        ensemble.eval()
        
        x = torch.randn(self.batch_size, self.input_channels, self.sequence_length)
        
        if hasattr(ensemble, 'predict_with_uncertainty'):
            predictions, uncertainties = ensemble.predict_with_uncertainty(x)
            
            assert predictions.shape == (self.batch_size, 1)
            assert uncertainties.shape == (self.batch_size, 1)
            assert torch.all(uncertainties >= 0)


class TestModelFactory:
    """Test suite for model factory functions."""
    
    def test_create_model_factory(self):
        """Test model creation factory."""
        model = create_model(
            input_channels=2,
            sequence_length=2048,
            model_config={'dropout_rate': 0.3}
        )
        
        assert isinstance(model, ExoplanetCNN)
        assert model.dropout_rate == 0.3
        
    def test_model_consistency(self):
        """Test consistency across model architectures."""
        input_shape = (4, 2, 2048)
        x = torch.randn(*input_shape)
        
        models = [
            ExoplanetCNN(2, 2048),
            ExoplanetLSTM(2, 2048),
            ExoplanetTransformer(2, 2048)
        ]
        
        for model in models:
            model.eval()
            output = model(x)
            
            # All models should produce same output shape
            assert output.shape == (4, 1)
            assert torch.all(output >= 0) and torch.all(output <= 1)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])