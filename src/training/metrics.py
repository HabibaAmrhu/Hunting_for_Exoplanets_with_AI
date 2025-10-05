"""
Comprehensive metrics calculation for exoplanet detection evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """
    Comprehensive metrics calculation and visualization for exoplanet detection.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'confusion_matrix': cm,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        # Probability-based metrics (if probabilities provided)
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
                
                metrics.update({
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                })
                
                # ROC and PR curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
                
                metrics.update({
                    'roc_curve': {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': roc_thresholds
                    },
                    'pr_curve': {
                        'precision': precision_curve,
                        'recall': recall_curve,
                        'thresholds': pr_thresholds
                    }
                })
                
            except ValueError as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                metrics.update({
                    'roc_auc': 0.0,
                    'pr_auc': 0.0
                })
        
        return metrics
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        metrics_to_compare: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_dict: Dictionary with model_name -> metrics_dict
            metrics_to_compare: List of metrics to include in comparison
            
        Returns:
            DataFrame with model comparison
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                'accuracy', 'precision', 'recall', 'f1_score', 
                'roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc'
            ]
        
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {'model': model_name}
            for metric in metrics_to_compare:
                row[metric] = metrics.get(metric, np.nan)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('model')
        
        return comparison_df
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = ['Non-Planet', 'Planet']
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                ax.text(
                    j + 0.5, i + 0.7, 
                    f'({percentage:.1f}%)', 
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(
        self,
        results_dict: Dict[str, Dict],
        title: str = "ROC Curves",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_dict: Dictionary with model_name -> metrics_dict
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model_name, metrics in results_dict.items():
            if 'roc_curve' in metrics:
                roc_data = metrics['roc_curve']
                auc_score = metrics.get('roc_auc', 0)
                
                ax.plot(
                    roc_data['fpr'], 
                    roc_data['tpr'],
                    label=f'{model_name} (AUC = {auc_score:.3f})',
                    linewidth=2
                )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        results_dict: Dict[str, Dict],
        title: str = "Precision-Recall Curves",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            results_dict: Dictionary with model_name -> metrics_dict
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model_name, metrics in results_dict.items():
            if 'pr_curve' in metrics:
                pr_data = metrics['pr_curve']
                pr_auc = metrics.get('pr_auc', 0)
                
                ax.plot(
                    pr_data['recall'], 
                    pr_data['precision'],
                    label=f'{model_name} (AP = {pr_auc:.3f})',
                    linewidth=2
                )
        
        # Calculate baseline (random classifier performance)
        # For imbalanced datasets, this is the positive class ratio
        if results_dict:
            first_metrics = next(iter(results_dict.values()))
            if 'tp' in first_metrics and 'fp' in first_metrics:
                total_positives = first_metrics['tp'] + first_metrics['fn']
                total_samples = (first_metrics['tp'] + first_metrics['tn'] + 
                               first_metrics['fp'] + first_metrics['fn'])
                baseline = total_positives / total_samples if total_samples > 0 else 0.5
                ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                          label=f'Random (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot training history curves.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score curves
        axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Training')
        axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Validation')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision curves
        axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Training')
        axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall curves
        axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='Training')
        axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> str:
        """
        Generate comprehensive classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_all_metrics(y_true, y_pred, y_prob)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"Classification Report: {model_name}")
        report_lines.append("=" * 60)
        
        # Basic metrics
        report_lines.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision:          {metrics['precision']:.4f}")
        report_lines.append(f"Recall:             {metrics['recall']:.4f}")
        report_lines.append(f"F1 Score:           {metrics['f1_score']:.4f}")
        report_lines.append(f"Specificity:        {metrics['specificity']:.4f}")
        report_lines.append(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        report_lines.append(f"MCC:                {metrics['mcc']:.4f}")
        
        if 'roc_auc' in metrics:
            report_lines.append(f"ROC AUC:            {metrics['roc_auc']:.4f}")
            report_lines.append(f"PR AUC:             {metrics['pr_auc']:.4f}")
        
        report_lines.append("-" * 60)
        report_lines.append("Confusion Matrix:")
        report_lines.append("-" * 60)
        report_lines.append(f"True Negatives:     {metrics['tn']}")
        report_lines.append(f"False Positives:    {metrics['fp']}")
        report_lines.append(f"False Negatives:    {metrics['fn']}")
        report_lines.append(f"True Positives:     {metrics['tp']}")
        
        # Class distribution
        total = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
        pos_ratio = (metrics['tp'] + metrics['fn']) / total if total > 0 else 0
        
        report_lines.append("-" * 60)
        report_lines.append("Dataset Information:")
        report_lines.append("-" * 60)
        report_lines.append(f"Total Samples:      {total}")
        report_lines.append(f"Positive Samples:   {metrics['tp'] + metrics['fn']} ({pos_ratio:.1%})")
        report_lines.append(f"Negative Samples:   {metrics['tn'] + metrics['fp']} ({1-pos_ratio:.1%})")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_metrics_to_json(
        self,
        metrics: Dict,
        save_path: Union[str, Path]
    ):
        """Save metrics dictionary to JSON file."""
        import json
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like roc_curve, pr_curve)
                json_metrics[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_metrics[key][subkey] = subvalue.tolist()
                    else:
                        json_metrics[key][subkey] = subvalue
            else:
                json_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Metrics saved to {save_path}")