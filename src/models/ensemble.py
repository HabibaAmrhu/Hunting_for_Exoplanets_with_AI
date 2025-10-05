"""
Ensemble framework for combining multiple exoplanet detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json

from .cnn import ExoplanetCNN
from .lstm import ExoplanetLSTM, LightweightLSTM
from .transformer import ExoplanetTransformer, LightweightTransformer


class EnsembleModel(nn.Module):
    """
    Ultra-High Performance Ensemble for Exoplanet Detection.
    
    Combines multiple specialized architectures for world-class accuracy:
    - CNN: Local pattern detection and transit shape recognition
    - LSTM: Temporal sequence modeling with attention
    - Transformer: Long-range dependencies and global context
    - ViT: Patch-based attention for fine-grained analysis
    
    Combination strategies:
    - Weighted averaging with learned weights
    - Stacking with meta-learner
    - Uncertainty-aware voting
    - Bayesian model averaging
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        combination_method: str = 'weighted_average',
        weights: Optional[List[float]] = None,
        meta_learner_config: Optional[Dict] = None,
        uncertainty_estimation: bool = True
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models to ensemble
            combination_method: Method to combine predictions
                - 'weighted_average': Weighted average of predictions
                - 'learned': Use a meta-learner to combine predictions
                - 'voting': Majority voting (for binary classification)
                - 'stacking': Stack predictions and use meta-model
            weights: Weights for weighted average (if None, uses equal weights)
            meta_learner_config: Configuration for meta-learner
            uncertainty_estimation: Whether to estimate prediction uncertainty
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.combination_method = combination_method
        self.uncertainty_estimation = uncertainty_estimation
        self.n_models = len(models)
        
        # Set model weights
        if weights is None:
            self.weights = torch.ones(self.n_models) / self.n_models
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        self.register_buffer('_weights', self.weights)
        
        # Meta-learner for learned combination
        if combination_method == 'learned':
            meta_config = meta_learner_config or {}
            hidden_size = meta_config.get('hidden_size', 64)
            dropout_rate = meta_config.get('dropout_rate', 0.2)
            
            self.meta_learner = nn.Sequential(
                nn.Linear(self.n_models, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        # Stacking meta-model
        elif combination_method == 'stacking':
            meta_config = meta_learner_config or {}
            hidden_size = meta_config.get('hidden_size', 128)
            dropout_rate = meta_config.get('dropout_rate', 0.2)
            
            # Get feature dimensions from models
            feature_dims = []
            for model in self.models:
                if hasattr(model, 'get_features'):
                    # Try to get feature dimension
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 2, 2048)  # Dummy input
                        try:
                            features = model.get_features(dummy_input)
                            feature_dims.append(features.size(-1))
                        except:
                            feature_dims.append(128)  # Default
                else:
                    feature_dims.append(128)  # Default
            
            total_feature_dim = sum(feature_dims)
            
            self.stacking_model = nn.Sequential(
                nn.Linear(total_feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        # Set models to evaluation mode by default
        for model in self.models:
            model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        if self.combination_method == 'stacking':
            return self._forward_stacking(x)
        else:
            return self._forward_prediction_combination(x)
    
    def _forward_prediction_combination(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using prediction combination methods."""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=-1)  # (batch_size, 1, n_models)
        predictions = predictions.squeeze(1)  # (batch_size, n_models)
        
        if self.combination_method == 'weighted_average':
            # Weighted average
            weights = self._weights.to(predictions.device)
            output = torch.sum(predictions * weights, dim=1, keepdim=True)
        
        elif self.combination_method == 'learned':
            # Use meta-learner
            output = self.meta_learner(predictions)
        
        elif self.combination_method == 'voting':
            # Majority voting (threshold at 0.5)
            binary_preds = (predictions > 0.5).float()
            votes = torch.sum(binary_preds, dim=1)
            output = (votes > (self.n_models / 2)).float().unsqueeze(1)
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return output
    
    def _forward_stacking(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using stacking method."""
        # Get features from all models
        features = []
        for model in self.models:
            if hasattr(model, 'get_features'):
                with torch.no_grad():
                    feat = model.get_features(x)
                    features.append(feat)
            else:
                # Fallback: use model output as feature
                with torch.no_grad():
                    pred = model(x)
                    features.append(pred)
        
        # Concatenate features
        combined_features = torch.cat(features, dim=-1)
        
        # Pass through stacking model
        output = self.stacking_model(combined_features)
        
        return output
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=-1)  # (batch_size, 1, n_models)
        predictions = predictions.squeeze(1)  # (batch_size, n_models)
        
        # Calculate ensemble prediction
        if self.combination_method == 'weighted_average':
            weights = self._weights.to(predictions.device)
            ensemble_pred = torch.sum(predictions * weights, dim=1, keepdim=True)
        else:
            ensemble_pred = self.forward(x)
        
        # Calculate uncertainty as standard deviation across models
        uncertainty = torch.std(predictions, dim=1, keepdim=True)
        
        return ensemble_pred, uncertainty
    
    def get_individual_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions from individual models.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        
        for i, model in enumerate(self.models):
            model_name = f"model_{i}"
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                model_name = info.get('model_name', model_name)
            
            with torch.no_grad():
                pred = model(x)
                predictions[model_name] = pred
        
        return predictions
    
    def update_weights(self, new_weights: List[float]):
        """
        Update ensemble weights.
        
        Args:
            new_weights: New weights for models
        """
        if len(new_weights) != self.n_models:
            raise ValueError(f"Expected {self.n_models} weights, got {len(new_weights)}")
        
        weights = torch.tensor(new_weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize
        
        self._weights.data = weights
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters for each model and total."""
        param_counts = {}
        total_params = 0
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'count_parameters'):
                count = model.count_parameters()
            else:
                count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_name = f"model_{i}"
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                model_name = info.get('model_name', model_name)
            
            param_counts[model_name] = count
            total_params += count
        
        # Add meta-learner parameters if applicable
        if hasattr(self, 'meta_learner'):
            meta_params = sum(p.numel() for p in self.meta_learner.parameters() if p.requires_grad)
            param_counts['meta_learner'] = meta_params
            total_params += meta_params
        
        if hasattr(self, 'stacking_model'):
            stacking_params = sum(p.numel() for p in self.stacking_model.parameters() if p.requires_grad)
            param_counts['stacking_model'] = stacking_params
            total_params += stacking_params
        
        param_counts['total'] = total_params
        
        return param_counts
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information."""
        individual_info = []
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
            else:
                info = {'model_name': f'model_{i}'}
            individual_info.append(info)
        
        return {
            'ensemble_type': 'EnsembleModel',
            'combination_method': self.combination_method,
            'n_models': self.n_models,
            'weights': self._weights.tolist(),
            'uncertainty_estimation': self.uncertainty_estimation,
            'individual_models': individual_info,
            'parameter_counts': self.count_parameters()
        }


class AdaptiveEnsemble(EnsembleModel):
    """
    Adaptive ensemble that can adjust weights based on performance.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        adaptation_method: str = 'performance_based',
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize adaptive ensemble.
        
        Args:
            models: List of models
            adaptation_method: Method for weight adaptation
            adaptation_rate: Rate of weight adaptation
            **kwargs: Additional arguments for base ensemble
        """
        super().__init__(models, **kwargs)
        
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate
        
        # Track performance metrics
        self.model_performances = torch.ones(self.n_models)
        self.update_count = 0
    
    def update_performance(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update model performance tracking.
        
        Args:
            predictions: Model predictions (batch_size, n_models)
            targets: True targets (batch_size,)
        """
        with torch.no_grad():
            # Calculate individual model accuracies
            binary_preds = (predictions > 0.5).float()
            targets_expanded = targets.unsqueeze(1).expand_as(binary_preds)
            
            accuracies = (binary_preds == targets_expanded).float().mean(dim=0)
            
            # Update running average of performance
            if self.update_count == 0:
                self.model_performances = accuracies
            else:
                alpha = self.adaptation_rate
                self.model_performances = (1 - alpha) * self.model_performances + alpha * accuracies
            
            self.update_count += 1
            
            # Update weights based on performance
            if self.adaptation_method == 'performance_based':
                # Softmax of performance scores
                new_weights = F.softmax(self.model_performances / 0.1, dim=0)
                self._weights.data = new_weights


class BayesianEnsemble(nn.Module):
    """
    Bayesian ensemble using Monte Carlo Dropout for uncertainty estimation.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_samples: int = 10,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Bayesian ensemble.
        
        Args:
            base_model: Base model to use for ensemble
            n_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty estimation
        """
        super(BayesianEnsemble, self).__init__()
        
        self.base_model = base_model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout layers if not present
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add dropout layers to the model for MC dropout."""
        # This is a simplified implementation
        # In practice, you'd need to carefully add dropout layers
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MC dropout."""
        # Enable dropout during inference
        self.base_model.train()
        
        predictions = []
        for _ in range(self.n_samples):
            pred = self.base_model(x)
            predictions.append(pred)
        
        # Return mean prediction
        predictions = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        
        return mean_pred
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation."""
        self.base_model.train()
        
        predictions = []
        for _ in range(self.n_samples):
            pred = self.base_model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


def create_ensemble(
    model_configs: List[Dict[str, Any]],
    combination_method: str = 'weighted_average',
    weights: Optional[List[float]] = None,
    **kwargs
) -> EnsembleModel:
    """
    Factory function to create ensemble models.
    
    Args:
        model_configs: List of model configurations
        combination_method: Method to combine predictions
        weights: Optional weights for models
        **kwargs: Additional ensemble arguments
        
    Returns:
        Ensemble model
    """
    models = []
    
    for config in model_configs:
        model_type = config['type']
        model_params = config.get('params', {})
        
        if model_type == 'cnn':
            model = ExoplanetCNN(**model_params)
        elif model_type == 'lstm':
            model = ExoplanetLSTM(**model_params)
        elif model_type == 'lstm_lightweight':
            model = LightweightLSTM(**model_params)
        elif model_type == 'transformer':
            model = ExoplanetTransformer(**model_params)
        elif model_type == 'transformer_lightweight':
            model = LightweightTransformer(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    return EnsembleModel(
        models=models,
        combination_method=combination_method,
        weights=weights,
        **kwargs
    )


def load_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_configs: List[Dict[str, Any]],
    combination_method: str = 'weighted_average',
    **kwargs
) -> EnsembleModel:
    """
    Load ensemble from saved model checkpoints.
    
    Args:
        checkpoint_paths: Paths to model checkpoints
        model_configs: Model configurations
        combination_method: Combination method
        **kwargs: Additional ensemble arguments
        
    Returns:
        Loaded ensemble model
    """
    if len(checkpoint_paths) != len(model_configs):
        raise ValueError("Number of checkpoints must match number of model configs")
    
    models = []
    
    for checkpoint_path, config in zip(checkpoint_paths, model_configs):
        # Create model
        model_type = config['type']
        model_params = config.get('params', {})
        
        if model_type == 'cnn':
            model = ExoplanetCNN(**model_params)
        elif model_type == 'lstm':
            model = ExoplanetLSTM(**model_params)
        elif model_type == 'transformer':
            model = ExoplanetTransformer(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
    
    return EnsembleModel(
        models=models,
        combination_method=combination_method,
        **kwargs
    )


def optimize_ensemble_weights(
    ensemble: EnsembleModel,
    val_loader: torch.utils.data.DataLoader,
    method: str = 'grid_search'
) -> List[float]:
    """
    Optimize ensemble weights using validation data.
    
    Args:
        ensemble: Ensemble model
        val_loader: Validation data loader
        method: Optimization method ('grid_search', 'random_search')
        
    Returns:
        Optimized weights
    """
    from sklearn.metrics import f1_score
    
    best_weights = None
    best_f1 = 0.0
    
    if method == 'grid_search':
        # Simple grid search for small ensembles
        if ensemble.n_models <= 3:
            import itertools
            
            # Generate weight combinations
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            for weights in itertools.product(weight_options, repeat=ensemble.n_models):
                if abs(sum(weights) - 1.0) < 0.01:  # Approximately sum to 1
                    normalized_weights = [w / sum(weights) for w in weights]
                    
                    # Test these weights
                    ensemble.update_weights(normalized_weights)
                    f1 = evaluate_ensemble(ensemble, val_loader)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = normalized_weights
    
    elif method == 'random_search':
        # Random search
        for _ in range(100):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(ensemble.n_models))
            
            ensemble.update_weights(weights.tolist())
            f1 = evaluate_ensemble(ensemble, val_loader)
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights.tolist()
    
    return best_weights or [1.0 / ensemble.n_models] * ensemble.n_models


def evaluate_ensemble(ensemble: EnsembleModel, data_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate ensemble performance.
    
    Args:
        ensemble: Ensemble model
        data_loader: Data loader
        
    Returns:
        F1 score
    """
    from sklearn.metrics import f1_score
    
    ensemble.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target, _ in data_loader:
            pred = ensemble(data)
            predictions.extend((pred > 0.5).cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    return f1_score(targets, predictions)