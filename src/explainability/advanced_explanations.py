"""
Advanced explainability methods for exoplanet detection models.
Extends basic Integrated Gradients with additional interpretation techniques.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

try:
    from captum.attr import (
        IntegratedGradients, GradientShap, DeepLift, DeepLiftShap,
        Saliency, InputXGradient, GuidedBackprop, GuidedGradCam
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class ExplainabilityMethod(ABC):
    """
    Abstract base class for explainability methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize explainability method.
        
        Args:
            model: Model to explain
            device: Device for computations
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    @abstractmethod
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate explanations for inputs.
        
        Args:
            inputs: Input tensor
            target: Target class (optional)
            **kwargs: Additional arguments
            
        Returns:
            Attribution tensor
        """
        pass
    
    def batch_explain(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate explanations for batch of inputs.
        
        Args:
            inputs: Batch of input tensors
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            Batch of attribution tensors
        """
        attributions = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_attr = self.explain(batch, **kwargs)
            attributions.append(batch_attr)
        
        return torch.cat(attributions, dim=0)


class IntegratedGradientsExplainer(ExplainabilityMethod):
    """
    Integrated Gradients explainer for time series data.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(model)
        else:
            self.ig = None
    
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baselines: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate Integrated Gradients explanations.
        
        Args:
            inputs: Input tensor
            target: Target class
            baselines: Baseline tensor
            n_steps: Number of integration steps
            **kwargs: Additional arguments
            
        Returns:
            Attribution tensor
        """
        if self.ig is None:
            return self._manual_integrated_gradients(
                inputs, target, baselines, n_steps
            )
        
        # Use Captum implementation
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        
        attributions = self.ig.attribute(
            inputs,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
            **kwargs
        )
        
        return attributions
    
    def _manual_integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: Optional[int],
        baselines: Optional[torch.Tensor],
        n_steps: int
    ) -> torch.Tensor:
        """
        Manual implementation of Integrated Gradients.
        """
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        integrated_grads = torch.zeros_like(inputs)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baselines + alpha * (inputs - baselines)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(interpolated)
            
            # Get target output
            if target is not None:
                target_output = outputs[:, target]
            else:
                target_output = outputs.squeeze()
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=target_output.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            integrated_grads += grads
        
        # Average and scale by input difference
        integrated_grads = integrated_grads / n_steps
        integrated_grads = integrated_grads * (inputs - baselines)
        
        return integrated_grads


class SHAPExplainer(ExplainabilityMethod):
    """
    SHAP-based explainer for model interpretability.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        
        if CAPTUM_AVAILABLE:
            self.gradient_shap = GradientShap(model)
        else:
            self.gradient_shap = None
    
    def explain(
        self,
        inputs: torch.Tensor,
        baselines: Optional[torch.Tensor] = None,
        n_samples: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate SHAP explanations.
        
        Args:
            inputs: Input tensor
            baselines: Baseline distribution
            n_samples: Number of samples for SHAP
            **kwargs: Additional arguments
            
        Returns:
            SHAP values
        """
        if self.gradient_shap is None:
            return self._manual_gradient_shap(inputs, baselines, n_samples)
        
        if baselines is None:
            baselines = torch.randn_like(inputs) * 0.1
        
        attributions = self.gradient_shap.attribute(
            inputs,
            baselines=baselines,
            n_samples=n_samples,
            **kwargs
        )
        
        return attributions
    
    def _manual_gradient_shap(
        self,
        inputs: torch.Tensor,
        baselines: Optional[torch.Tensor],
        n_samples: int
    ) -> torch.Tensor:
        """
        Manual implementation of Gradient SHAP.
        """
        if baselines is None:
            baselines = torch.randn_like(inputs) * 0.1
        
        attributions = torch.zeros_like(inputs)
        
        for _ in range(n_samples):
            # Sample random baseline
            if baselines.dim() > inputs.dim():
                baseline = baselines[torch.randint(0, len(baselines), (1,))]
            else:
                baseline = baselines
            
            # Random interpolation coefficient
            alpha = torch.rand(1, device=self.device)
            
            # Interpolated input
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(interpolated)
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=outputs.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Accumulate attributions
            attributions += grads * (inputs - baseline)
        
        return attributions / n_samples


class LIMEExplainer(ExplainabilityMethod):
    """
    LIME-inspired explainer for time series data.
    """
    
    def explain(
        self,
        inputs: torch.Tensor,
        n_samples: int = 1000,
        feature_selection: str = 'auto',
        **kwargs
    ) -> torch.Tensor:
        """
        Generate LIME explanations for time series.
        
        Args:
            inputs: Input tensor
            n_samples: Number of perturbation samples
            feature_selection: Feature selection method
            **kwargs: Additional arguments
            
        Returns:
            Feature importance scores
        """
        batch_size, seq_len = inputs.shape[:2]
        
        # Generate perturbations by masking segments
        segment_size = max(1, seq_len // 20)  # 20 segments
        n_segments = seq_len // segment_size
        
        # Create perturbation matrix
        perturbations = torch.randint(
            0, 2, (n_samples, n_segments), 
            device=self.device, dtype=torch.float32
        )
        
        # Generate perturbed samples
        perturbed_inputs = []
        for i in range(n_samples):
            perturbed = inputs.clone()
            for j in range(n_segments):
                if perturbations[i, j] == 0:  # Mask this segment
                    start_idx = j * segment_size
                    end_idx = min((j + 1) * segment_size, seq_len)
                    perturbed[:, start_idx:end_idx] = 0
            perturbed_inputs.append(perturbed)
        
        perturbed_batch = torch.stack(perturbed_inputs)
        
        # Get predictions for perturbed samples
        with torch.no_grad():
            original_pred = self.model(inputs)
            perturbed_preds = []
            
            for i in range(0, n_samples, 32):  # Process in batches
                batch_end = min(i + 32, n_samples)
                batch_perturbed = perturbed_batch[i:batch_end]
                batch_preds = self.model(batch_perturbed.view(-1, *inputs.shape[1:]))
                perturbed_preds.append(batch_preds)
            
            perturbed_preds = torch.cat(perturbed_preds, dim=0)
        
        # Fit linear model to explain predictions
        attributions = self._fit_linear_model(
            perturbations, perturbed_preds, original_pred, segment_size, seq_len
        )
        
        return attributions
    
    def _fit_linear_model(
        self,
        perturbations: torch.Tensor,
        predictions: torch.Tensor,
        original_pred: torch.Tensor,
        segment_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Fit linear model to get feature importances.
        """
        # Simple linear regression using least squares
        X = perturbations.cpu().numpy()
        y = (predictions - original_pred).cpu().numpy().flatten()
        
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve least squares
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            segment_importances = coefficients[1:]  # Exclude intercept
        except np.linalg.LinAlgError:
            segment_importances = np.zeros(perturbations.shape[1])
        
        # Map segment importances back to time series
        attributions = torch.zeros(seq_len, device=self.device)
        n_segments = len(segment_importances)
        
        for i, importance in enumerate(segment_importances):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, seq_len)
            attributions[start_idx:end_idx] = importance
        
        return attributions.unsqueeze(0)  # Add batch dimension


class AdvancedExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis combining multiple methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize advanced explainability analyzer.
        
        Args:
            model: Model to analyze
            device: Device for computations
        """
        self.model = model
        self.device = device
        
        # Initialize explainers
        self.explainers = {
            'integrated_gradients': IntegratedGradientsExplainer(model, device),
            'shap': SHAPExplainer(model, device),
            'lime': LIMEExplainer(model, device)
        }
    
    def comprehensive_analysis(
        self,
        inputs: torch.Tensor,
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Run comprehensive explainability analysis.
        
        Args:
            inputs: Input tensor
            methods: List of methods to use
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of explanations from different methods
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        explanations = {}
        
        for method in methods:
            if method in self.explainers:
                try:
                    explanations[method] = self.explainers[method].explain(
                        inputs, **kwargs
                    )
                except Exception as e:
                    print(f"Warning: {method} failed with error: {e}")
                    explanations[method] = torch.zeros_like(inputs)
        
        return explanations
    
    def plot_explanations(
        self,
        inputs: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        save_path: Optional[Path] = None,
        sample_idx: int = 0
    ):
        """
        Plot explanations from different methods.
        
        Args:
            inputs: Original input tensor
            explanations: Dictionary of explanations
            save_path: Path to save plot
            sample_idx: Index of sample to plot
        """
        n_methods = len(explanations)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(12, 3 * (n_methods + 1)))
        
        if n_methods == 0:
            axes = [axes]
        
        # Plot original input
        time_steps = range(len(inputs[sample_idx]))
        axes[0].plot(time_steps, inputs[sample_idx].cpu().numpy(), 'b-', linewidth=2)
        axes[0].set_title('Original Light Curve')
        axes[0].set_ylabel('Flux')
        axes[0].grid(True, alpha=0.3)
        
        # Plot explanations
        for i, (method, attribution) in enumerate(explanations.items()):
            attr_values = attribution[sample_idx].cpu().numpy()
            
            # Normalize attribution values
            if attr_values.max() != attr_values.min():
                attr_values = (attr_values - attr_values.min()) / (attr_values.max() - attr_values.min())
            
            axes[i + 1].plot(time_steps, attr_values, 'r-', linewidth=2)
            axes[i + 1].set_title(f'{method.replace("_", " ").title()} Attribution')
            axes[i + 1].set_ylabel('Attribution')
            axes[i + 1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def feature_importance_summary(
        self,
        explanations: Dict[str, torch.Tensor],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Generate feature importance summary.
        
        Args:
            explanations: Dictionary of explanations
            top_k: Number of top features to return
            
        Returns:
            Dictionary of top important features for each method
        """
        summary = {}
        
        for method, attribution in explanations.items():
            # Average across batch dimension
            avg_attribution = attribution.mean(dim=0).abs()
            
            # Get top-k features
            top_values, top_indices = torch.topk(avg_attribution, top_k)
            
            summary[method] = [
                (idx.item(), val.item()) 
                for idx, val in zip(top_indices, top_values)
            ]
        
        return summary


def create_explainability_analyzer(
    model: nn.Module, 
    device: torch.device
) -> AdvancedExplainabilityAnalyzer:
    """
    Factory function to create explainability analyzer.
    
    Args:
        model: Model to analyze
        device: Device for computations
        
    Returns:
        Configured explainability analyzer
    """
    return AdvancedExplainabilityAnalyzer(model, device)