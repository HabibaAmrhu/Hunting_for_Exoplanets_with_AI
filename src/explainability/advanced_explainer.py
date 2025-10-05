"""
Advanced explainability features for exoplanet detection models.
Implements SHAP, counterfactual explanations, and interactive dashboards.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    
    method: str
    attributions: np.ndarray
    baseline: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'attributions': self.attributions.tolist() if self.attributions is not None else None,
            'baseline': self.baseline.tolist() if self.baseline is not None else None,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) explainer for time series models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: PyTorch model to explain
            background_data: Background dataset for SHAP
            device: Device for computations
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare background data
        if background_data is not None:
            self.background_data = background_data.to(self.device)
        else:
            # Create default background (zeros)
            self.background_data = torch.zeros(10, 2, 2048, device=self.device)
        
        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(self._model_wrapper, self.background_data)
    
    def _model_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper for model to ensure proper output format."""
        with torch.no_grad():
            output = self.model(x)
            return output.squeeze()
    
    def explain(
        self,
        input_data: torch.Tensor,
        n_samples: int = 100
    ) -> ExplanationResult:
        """
        Generate SHAP explanations.
        
        Args:
            input_data: Input data to explain
            n_samples: Number of samples for SHAP estimation
            
        Returns:
            Explanation result
        """
        input_data = input_data.to(self.device)
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(input_data, nsamples=n_samples)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(input_data).squeeze().cpu().numpy()
        
        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        
        return ExplanationResult(
            method="SHAP",
            attributions=shap_values,
            confidence=float(prediction) if prediction.ndim == 0 else prediction,
            metadata={
                'n_samples': n_samples,
                'background_size': len(self.background_data)
            }
        )


class CounterfactualExplainer:
    """
    Counterfactual explanation generator for exoplanet detection.
    
    Generates minimal changes to input that would flip the prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize counterfactual explainer.
        
        Args:
            model: PyTorch model
            device: Device for computations
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def generate_counterfactual(
        self,
        input_data: torch.Tensor,
        target_class: int,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate counterfactual explanation.
        
        Args:
            input_data: Original input data
            target_class: Target class for counterfactual
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            lambda_reg: Regularization strength
            
        Returns:
            Tuple of (counterfactual_data, metadata)
        """
        input_data = input_data.to(self.device)
        original_input = input_data.clone()
        
        # Initialize counterfactual as copy of original
        counterfactual = input_data.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([counterfactual], lr=learning_rate)
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(original_input).squeeze()
        
        best_counterfactual = None
        best_distance = float('inf')
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(counterfactual).squeeze()
            
            # Classification loss (encourage target class)
            if target_class == 1:
                class_loss = -torch.log(pred + 1e-8)
            else:
                class_loss = -torch.log(1 - pred + 1e-8)
            
            # Distance loss (minimize change from original)
            distance_loss = torch.norm(counterfactual - original_input, p=2)
            
            # Total loss
            total_loss = class_loss + lambda_reg * distance_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Check if we've achieved the target class
            current_pred = pred.item()
            target_achieved = (
                (target_class == 1 and current_pred > 0.5) or
                (target_class == 0 and current_pred < 0.5)
            )
            
            if target_achieved:
                current_distance = distance_loss.item()
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_counterfactual = counterfactual.clone().detach()
        
        if best_counterfactual is None:
            best_counterfactual = counterfactual.detach()
        
        # Calculate final metrics
        with torch.no_grad():
            final_pred = self.model(best_counterfactual).squeeze()
            distance = torch.norm(best_counterfactual - original_input, p=2).item()
        
        metadata = {
            'original_prediction': original_pred.item(),
            'counterfactual_prediction': final_pred.item(),
            'distance': distance,
            'iterations': max_iterations,
            'target_achieved': (
                (target_class == 1 and final_pred.item() > 0.5) or
                (target_class == 0 and final_pred.item() < 0.5)
            )
        }
        
        return best_counterfactual, metadata
    
    def explain(
        self,
        input_data: torch.Tensor,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate counterfactual explanation.
        
        Args:
            input_data: Input data to explain
            **kwargs: Additional arguments for counterfactual generation
            
        Returns:
            Explanation result
        """
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(input_data).squeeze()
        
        # Determine target class (opposite of current prediction)
        target_class = 0 if original_pred > 0.5 else 1
        
        # Generate counterfactual
        counterfactual, metadata = self.generate_counterfactual(
            input_data, target_class, **kwargs
        )
        
        # Calculate attribution as difference
        attribution = (counterfactual - input_data).cpu().numpy()
        
        return ExplanationResult(
            method="Counterfactual",
            attributions=attribution,
            baseline=input_data.cpu().numpy(),
            confidence=metadata['original_prediction'],
            metadata=metadata
        )


class FeatureImportanceAnalyzer:
    """
    Analyzer for feature importance across different time windows.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: PyTorch model
            device: Device for computations
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def analyze_temporal_importance(
        self,
        input_data: torch.Tensor,
        window_size: int = 64,
        stride: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance across temporal windows.
        
        Args:
            input_data: Input data to analyze
            window_size: Size of temporal windows
            stride: Stride between windows
            
        Returns:
            Dictionary with importance scores for each window
        """
        input_data = input_data.to(self.device)
        sequence_length = input_data.shape[-1]
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(input_data).squeeze()
        
        window_importances = []
        window_positions = []
        
        # Analyze each window
        for start in range(0, sequence_length - window_size + 1, stride):
            end = start + window_size
            
            # Create masked version (zero out the window)
            masked_input = input_data.clone()
            masked_input[:, :, start:end] = 0
            
            # Get prediction with masked input
            with torch.no_grad():
                masked_pred = self.model(masked_input).squeeze()
            
            # Calculate importance as difference in prediction
            importance = abs(baseline_pred - masked_pred).item()
            
            window_importances.append(importance)
            window_positions.append((start, end))
        
        return {
            'importances': np.array(window_importances),
            'positions': window_positions,
            'baseline_prediction': baseline_pred.item()
        }
    
    def analyze_channel_importance(
        self,
        input_data: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze importance of different input channels.
        
        Args:
            input_data: Input data to analyze
            
        Returns:
            Dictionary with importance scores for each channel
        """
        input_data = input_data.to(self.device)
        n_channels = input_data.shape[1]
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(input_data).squeeze()
        
        channel_importances = {}
        
        for channel in range(n_channels):
            # Create version with channel zeroed out
            masked_input = input_data.clone()
            masked_input[:, channel, :] = 0
            
            # Get prediction
            with torch.no_grad():
                masked_pred = self.model(masked_input).squeeze()
            
            # Calculate importance
            importance = abs(baseline_pred - masked_pred).item()
            channel_importances[f'channel_{channel}'] = importance
        
        return channel_importances


class InteractiveExplanationDashboard:
    """
    Interactive dashboard for model explanations.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.explanations = []
    
    def add_explanation(self, explanation: ExplanationResult, input_data: np.ndarray):
        """
        Add explanation to dashboard.
        
        Args:
            explanation: Explanation result
            input_data: Original input data
        """
        self.explanations.append({
            'explanation': explanation,
            'input_data': input_data,
            'timestamp': pd.Timestamp.now()
        })
    
    def create_explanation_plot(
        self,
        explanation: ExplanationResult,
        input_data: np.ndarray,
        title: str = "Model Explanation"
    ) -> go.Figure:
        """
        Create interactive explanation plot.
        
        Args:
            explanation: Explanation result
            input_data: Original input data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Raw Light Curve', 'Phase-Folded Light Curve',
                'Raw Attribution', 'Phase-Folded Attribution',
                'Combined View', 'Explanation Summary'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None]
            ]
        )
        
        # Ensure input_data has correct shape
        if input_data.ndim == 3:
            input_data = input_data[0]  # Remove batch dimension
        
        # Plot original light curves
        time_axis = np.arange(input_data.shape[1])
        
        # Raw channel
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=input_data[0],
                mode='lines',
                name='Raw',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Phase-folded channel
        if input_data.shape[0] > 1:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=input_data[1],
                    mode='lines',
                    name='Phase-folded',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # Plot attributions
        if explanation.attributions is not None:
            attributions = explanation.attributions
            if attributions.ndim == 3:
                attributions = attributions[0]  # Remove batch dimension
            
            # Raw attribution
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=attributions[0],
                    mode='lines',
                    name='Raw Attribution',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Phase-folded attribution
            if attributions.shape[0] > 1:
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=attributions[1],
                        mode='lines',
                        name='Phase-folded Attribution',
                        line=dict(color='orange')
                    ),
                    row=2, col=2
                )
            
            # Combined view with attribution overlay
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=input_data[0],
                    mode='lines',
                    name='Raw Data',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            # Add attribution as background color
            positive_attr = np.maximum(attributions[0], 0)
            negative_attr = np.minimum(attributions[0], 0)
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=positive_attr,
                    fill='tozeroy',
                    mode='none',
                    name='Positive Attribution',
                    fillcolor='rgba(255, 0, 0, 0.3)'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=negative_attr,
                    fill='tozeroy',
                    mode='none',
                    name='Negative Attribution',
                    fillcolor='rgba(0, 0, 255, 0.3)'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{title} - Method: {explanation.method}",
            height=800,
            showlegend=True
        )
        
        # Add confidence annotation
        if explanation.confidence is not None:
            fig.add_annotation(
                text=f"Confidence: {explanation.confidence:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        
        return fig
    
    def create_comparison_plot(
        self,
        explanations: List[ExplanationResult],
        input_data: np.ndarray,
        methods: List[str]
    ) -> go.Figure:
        """
        Create comparison plot for multiple explanation methods.
        
        Args:
            explanations: List of explanation results
            input_data: Original input data
            methods: List of method names
            
        Returns:
            Plotly figure comparing different methods
        """
        fig = make_subplots(
            rows=len(explanations) + 1, cols=1,
            subplot_titles=['Original Data'] + [f'{method} Attribution' for method in methods]
        )
        
        time_axis = np.arange(input_data.shape[-1])
        
        # Plot original data
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=input_data[0, 0] if input_data.ndim == 3 else input_data[0],
                mode='lines',
                name='Original',
                line=dict(color='black')
            ),
            row=1, col=1
        )
        
        # Plot each explanation
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (explanation, method) in enumerate(zip(explanations, methods)):
            if explanation.attributions is not None:
                attributions = explanation.attributions
                if attributions.ndim == 3:
                    attributions = attributions[0, 0]  # First batch, first channel
                elif attributions.ndim == 2:
                    attributions = attributions[0]  # First channel
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=attributions,
                        mode='lines',
                        name=f'{method} Attribution',
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=i + 2, col=1
                )
        
        fig.update_layout(
            title="Explanation Method Comparison",
            height=200 * (len(explanations) + 1),
            showlegend=True
        )
        
        return fig


# Factory functions
def create_shap_explainer(
    model: nn.Module,
    background_data: Optional[torch.Tensor] = None,
    **kwargs
) -> SHAPExplainer:
    """Create SHAP explainer."""
    return SHAPExplainer(model, background_data, **kwargs)


def create_counterfactual_explainer(
    model: nn.Module,
    **kwargs
) -> CounterfactualExplainer:
    """Create counterfactual explainer."""
    return CounterfactualExplainer(model, **kwargs)


def create_feature_importance_analyzer(
    model: nn.Module,
    **kwargs
) -> FeatureImportanceAnalyzer:
    """Create feature importance analyzer."""
    return FeatureImportanceAnalyzer(model, **kwargs)


def create_explanation_dashboard() -> InteractiveExplanationDashboard:
    """Create interactive explanation dashboard."""
    return InteractiveExplanationDashboard()