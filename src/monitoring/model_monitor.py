"""
Automated model monitoring and alerting system for exoplanet detection pipeline.
Provides drift detection, performance monitoring, and automated alerts.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings

try:
    from scipy import stats
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy/sklearn not available. Some monitoring features will be limited.")


@dataclass
class MonitoringMetrics:
    """
    Container for monitoring metrics.
    """
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_confidence: float
    data_drift_score: float
    model_drift_score: float
    throughput: float  # predictions per second
    latency: float  # average prediction time in ms
    memory_usage: float  # MB
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringMetrics':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AlertThresholds:
    """
    Thresholds for triggering alerts.
    """
    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1_score: float = 0.80
    max_data_drift: float = 0.3
    max_model_drift: float = 0.3
    min_throughput: float = 10.0  # predictions per second
    max_latency: float = 1000.0  # milliseconds
    max_memory_usage: float = 2048.0  # MB
    max_error_rate: float = 0.05


class DriftDetector(ABC):
    """
    Abstract base class for drift detection methods.
    """
    
    @abstractmethod
    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Detect drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Tuple of (drift_score, is_drift_detected)
        """
        pass


class KSTestDriftDetector(DriftDetector):
    """
    Kolmogorov-Smirnov test for drift detection.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize KS test drift detector.
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
    
    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Tuple of (drift_score, is_drift_detected)
        """
        if not SCIPY_AVAILABLE:
            return 0.0, False
        
        # Flatten arrays if multidimensional
        ref_flat = reference_data.flatten()
        cur_flat = current_data.flatten()
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(ref_flat, cur_flat)
        
        # Drift detected if p-value is below significance level
        is_drift = p_value < self.significance_level
        
        return statistic, is_drift


class PSIDriftDetector(DriftDetector):
    """
    Population Stability Index (PSI) for drift detection.
    """
    
    def __init__(self, n_bins: int = 10, threshold: float = 0.2):
        """
        Initialize PSI drift detector.
        
        Args:
            n_bins: Number of bins for histogram
            threshold: PSI threshold for drift detection
        """
        self.n_bins = n_bins
        self.threshold = threshold
    
    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Detect drift using Population Stability Index.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Tuple of (psi_score, is_drift_detected)
        """
        # Flatten arrays if multidimensional
        ref_flat = reference_data.flatten()
        cur_flat = current_data.flatten()
        
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(ref_flat, bins=self.n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_flat, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(cur_flat, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        ref_hist = ref_hist + epsilon
        cur_hist = cur_hist + epsilon
        
        # Calculate PSI
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        
        is_drift = psi > self.threshold
        
        return psi, is_drift


class ModelPerformanceMonitor:
    """
    Monitor model performance metrics over time.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        reference_data: Optional[np.ndarray] = None,
        drift_detector: Optional[DriftDetector] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            model: Model to monitor
            device: Device for inference
            reference_data: Reference dataset for drift detection
            drift_detector: Drift detection method
        """
        self.model = model
        self.device = device
        self.reference_data = reference_data
        self.drift_detector = drift_detector or KSTestDriftDetector()
        
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[MonitoringMetrics] = []
        
        # Performance tracking
        self.prediction_times = []
        self.error_count = 0
        self.total_predictions = 0
    
    def evaluate_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        return_predictions: bool = False
    ) -> Tuple[MonitoringMetrics, Optional[torch.Tensor]]:
        """
        Evaluate model performance on a batch.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            return_predictions: Whether to return predictions
            
        Returns:
            Tuple of (monitoring_metrics, predictions)
        """
        start_time = time.time()
        
        try:
            # Model inference
            with torch.no_grad():
                predictions = self.model(inputs.to(self.device))
                
            # Convert to numpy for metrics calculation
            pred_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Binary classification metrics
            pred_binary = (pred_np > 0.5).astype(int)
            
            if SCIPY_AVAILABLE:
                accuracy = accuracy_score(targets_np, pred_binary)
                precision = precision_score(targets_np, pred_binary, zero_division=0)
                recall = recall_score(targets_np, pred_binary, zero_division=0)
                f1 = f1_score(targets_np, pred_binary, zero_division=0)
            else:
                # Manual calculation
                tp = np.sum((pred_binary == 1) & (targets_np == 1))
                fp = np.sum((pred_binary == 1) & (targets_np == 0))
                tn = np.sum((pred_binary == 0) & (targets_np == 0))
                fn = np.sum((pred_binary == 0) & (targets_np == 1))
                
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Prediction confidence
            confidence = np.mean(np.abs(pred_np - 0.5) + 0.5)
            
            # Drift detection
            data_drift_score = 0.0
            model_drift_score = 0.0
            
            if self.reference_data is not None:
                try:
                    data_drift_score, _ = self.drift_detector.detect_drift(
                        self.reference_data, inputs.cpu().numpy()
                    )
                except Exception as e:
                    self.logger.warning(f"Data drift detection failed: {e}")
            
            # Performance metrics
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            throughput = len(inputs) / (end_time - start_time)
            
            # Memory usage (rough estimate)
            memory_usage = torch.cuda.memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0
            
            # Error rate
            self.total_predictions += len(inputs)
            error_rate = self.error_count / self.total_predictions if self.total_predictions > 0 else 0
            
            # Create monitoring metrics
            metrics = MonitoringMetrics(
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                prediction_confidence=confidence,
                data_drift_score=data_drift_score,
                model_drift_score=model_drift_score,
                throughput=throughput,
                latency=latency,
                memory_usage=memory_usage,
                error_rate=error_rate
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.prediction_times.append(latency)
            
            return metrics, predictions if return_predictions else None
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Evaluation failed: {e}")
            
            # Return default metrics on error
            error_metrics = MonitoringMetrics(
                timestamp=datetime.now(),
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                prediction_confidence=0.0, data_drift_score=1.0, model_drift_score=1.0,
                throughput=0.0, latency=float('inf'), memory_usage=0.0, error_rate=1.0
            )
            
            return error_metrics, None
    
    def get_recent_metrics(self, hours: int = 24) -> List[MonitoringMetrics]:
        """
        Get metrics from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, float]:
        """
        Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Dictionary with performance summary
        """
        recent_metrics = self.get_recent_metrics(hours)
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_accuracy': np.mean([m.accuracy for m in recent_metrics]),
            'avg_precision': np.mean([m.precision for m in recent_metrics]),
            'avg_recall': np.mean([m.recall for m in recent_metrics]),
            'avg_f1_score': np.mean([m.f1_score for m in recent_metrics]),
            'avg_confidence': np.mean([m.prediction_confidence for m in recent_metrics]),
            'max_data_drift': max([m.data_drift_score for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'avg_latency': np.mean([m.latency for m in recent_metrics]),
            'max_memory_usage': max([m.memory_usage for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics])
        }


class AlertManager:
    """
    Manage alerts and notifications for model monitoring.
    """
    
    def __init__(
        self,
        thresholds: AlertThresholds,
        email_config: Optional[Dict[str, str]] = None,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            thresholds: Alert thresholds
            email_config: Email configuration
            webhook_url: Webhook URL for notifications
        """
        self.thresholds = thresholds
        self.email_config = email_config
        self.webhook_url = webhook_url
        
        self.logger = logging.getLogger(__name__)
        self.alert_history: List[Dict[str, Any]] = []
        
        # Cooldown to prevent spam
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(minutes=30)
    
    def check_alerts(self, metrics: MonitoringMetrics) -> List[Dict[str, Any]]:
        """
        Check if any alerts should be triggered.
        
        Args:
            metrics: Current monitoring metrics
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Performance alerts
        if metrics.accuracy < self.thresholds.min_accuracy:
            alerts.append({
                'type': 'performance',
                'metric': 'accuracy',
                'value': metrics.accuracy,
                'threshold': self.thresholds.min_accuracy,
                'severity': 'high'
            })
        
        if metrics.precision < self.thresholds.min_precision:
            alerts.append({
                'type': 'performance',
                'metric': 'precision',
                'value': metrics.precision,
                'threshold': self.thresholds.min_precision,
                'severity': 'medium'
            })
        
        if metrics.recall < self.thresholds.min_recall:
            alerts.append({
                'type': 'performance',
                'metric': 'recall',
                'value': metrics.recall,
                'threshold': self.thresholds.min_recall,
                'severity': 'medium'
            })
        
        if metrics.f1_score < self.thresholds.min_f1_score:
            alerts.append({
                'type': 'performance',
                'metric': 'f1_score',
                'value': metrics.f1_score,
                'threshold': self.thresholds.min_f1_score,
                'severity': 'medium'
            })
        
        # Drift alerts
        if metrics.data_drift_score > self.thresholds.max_data_drift:
            alerts.append({
                'type': 'drift',
                'metric': 'data_drift',
                'value': metrics.data_drift_score,
                'threshold': self.thresholds.max_data_drift,
                'severity': 'high'
            })
        
        if metrics.model_drift_score > self.thresholds.max_model_drift:
            alerts.append({
                'type': 'drift',
                'metric': 'model_drift',
                'value': metrics.model_drift_score,
                'threshold': self.thresholds.max_model_drift,
                'severity': 'high'
            })
        
        # System alerts
        if metrics.throughput < self.thresholds.min_throughput:
            alerts.append({
                'type': 'system',
                'metric': 'throughput',
                'value': metrics.throughput,
                'threshold': self.thresholds.min_throughput,
                'severity': 'medium'
            })
        
        if metrics.latency > self.thresholds.max_latency:
            alerts.append({
                'type': 'system',
                'metric': 'latency',
                'value': metrics.latency,
                'threshold': self.thresholds.max_latency,
                'severity': 'medium'
            })
        
        if metrics.memory_usage > self.thresholds.max_memory_usage:
            alerts.append({
                'type': 'system',
                'metric': 'memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.thresholds.max_memory_usage,
                'severity': 'low'
            })
        
        if metrics.error_rate > self.thresholds.max_error_rate:
            alerts.append({
                'type': 'system',
                'metric': 'error_rate',
                'value': metrics.error_rate,
                'threshold': self.thresholds.max_error_rate,
                'severity': 'high'
            })
        
        # Add timestamp and send notifications
        for alert in alerts:
            alert['timestamp'] = metrics.timestamp
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """
        Send alert notification.
        
        Args:
            alert: Alert information
        """
        alert_key = f"{alert['type']}_{alert['metric']}"
        
        # Check cooldown
        if alert_key in self.last_alert_time:
            if datetime.now() - self.last_alert_time[alert_key] < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = datetime.now()
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            f"ALERT: {alert['type'].upper()} - {alert['metric']} = {alert['value']:.4f} "
            f"(threshold: {alert['threshold']:.4f}, severity: {alert['severity']})"
        )
        
        # Send email notification
        if self.email_config:
            self._send_email_alert(alert)
        
        # Send webhook notification
        if self.webhook_url:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Model Alert: {alert['type'].title()} Issue Detected"
            
            body = f"""
            Alert Details:
            - Type: {alert['type'].title()}
            - Metric: {alert['metric']}
            - Current Value: {alert['value']:.4f}
            - Threshold: {alert['threshold']:.4f}
            - Severity: {alert['severity'].title()}
            - Timestamp: {alert['timestamp']}
            
            Please investigate the issue promptly.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                'alert_type': alert['type'],
                'metric': alert['metric'],
                'value': alert['value'],
                'threshold': alert['threshold'],
                'severity': alert['severity'],
                'timestamp': alert['timestamp'].isoformat()
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")


class ModelMonitoringSystem:
    """
    Complete model monitoring system.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        thresholds: Optional[AlertThresholds] = None,
        reference_data: Optional[np.ndarray] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize monitoring system.
        
        Args:
            model: Model to monitor
            device: Device for inference
            thresholds: Alert thresholds
            reference_data: Reference data for drift detection
            storage_path: Path to store monitoring data
        """
        self.model = model
        self.device = device
        self.thresholds = thresholds or AlertThresholds()
        self.storage_path = storage_path or Path('monitoring_data')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.performance_monitor = ModelPerformanceMonitor(
            model, device, reference_data
        )
        self.alert_manager = AlertManager(self.thresholds)
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Monitor a batch of predictions.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            Monitoring results
        """
        # Evaluate performance
        metrics, predictions = self.performance_monitor.evaluate_batch(
            inputs, targets, return_predictions=True
        )
        
        # Check for alerts
        alerts = self.alert_manager.check_alerts(metrics)
        
        # Save metrics
        self._save_metrics(metrics)
        
        return {
            'metrics': metrics,
            'alerts': alerts,
            'predictions': predictions
        }
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.
        
        Args:
            hours: Hours of data to include
            
        Returns:
            Dashboard data
        """
        recent_metrics = self.performance_monitor.get_recent_metrics(hours)
        summary = self.performance_monitor.get_performance_summary(hours)
        recent_alerts = [
            alert for alert in self.alert_manager.alert_history
            if alert['timestamp'] >= datetime.now() - timedelta(hours=hours)
        ]
        
        return {
            'summary': summary,
            'metrics_history': [m.to_dict() for m in recent_metrics],
            'recent_alerts': recent_alerts,
            'thresholds': asdict(self.thresholds)
        }
    
    def _save_metrics(self, metrics: MonitoringMetrics):
        """Save metrics to storage."""
        try:
            metrics_file = self.storage_path / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")


def create_monitoring_system(
    model: torch.nn.Module,
    device: torch.device,
    thresholds: Optional[AlertThresholds] = None,
    reference_data: Optional[np.ndarray] = None,
    storage_path: Optional[Path] = None
) -> ModelMonitoringSystem:
    """
    Factory function to create monitoring system.
    
    Args:
        model: Model to monitor
        device: Device for inference
        thresholds: Alert thresholds
        reference_data: Reference data for drift detection
        storage_path: Path to store monitoring data
        
    Returns:
        Configured monitoring system
    """
    return ModelMonitoringSystem(
        model, device, thresholds, reference_data, storage_path
    )