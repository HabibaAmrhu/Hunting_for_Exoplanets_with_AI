"""
Real-time processing pipeline for exoplanet detection.
Implements streaming data ingestion and real-time model inference.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import queue
import threading

from ..data.types import LightCurve, ProcessedLightCurve
from ..preprocessing.preprocessor import LightCurvePreprocessor
from ..models.cnn import ExoplanetCNN


@dataclass
class DetectionAlert:
    """Alert for high-confidence exoplanet detection."""
    
    star_id: str
    confidence: float
    timestamp: datetime
    light_curve_data: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'star_id': self.star_id,
            'confidence': float(self.confidence),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


class RealTimeProcessor:
    """
    Real-time processor for streaming exoplanet detection.
    
    Features:
    - Asynchronous data ingestion
    - Real-time model inference
    - Alert generation for high-confidence detections
    - Performance monitoring and logging
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: LightCurvePreprocessor,
        confidence_threshold: float = 0.8,
        alert_callback: Optional[Callable[[DetectionAlert], None]] = None,
        max_queue_size: int = 1000,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize real-time processor.
        
        Args:
            model: Trained exoplanet detection model
            preprocessor: Light curve preprocessor
            confidence_threshold: Threshold for generating alerts
            alert_callback: Callback function for handling alerts
            max_queue_size: Maximum size of processing queue
            batch_size: Batch size for inference
            device: Device for model inference
        """
        self.model = model
        self.preprocessor = preprocessor
        self.confidence_threshold = confidence_threshold
        self.alert_callback = alert_callback
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Processing queue
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_stats = {
            'processed_count': 0,
            'alert_count': 0,
            'processing_time_total': 0.0,
            'start_time': time.time()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Processing task
        self.processing_task = None
        self.is_running = False
    
    async def start(self):
        """Start the real-time processing pipeline."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.logger.info("Real-time processor started")
    
    async def stop(self):
        """Stop the real-time processing pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Real-time processor stopped")
    
    async def ingest_light_curve(self, light_curve: LightCurve):
        """
        Ingest a light curve for processing.
        
        Args:
            light_curve: Light curve to process
        """
        try:
            await self.input_queue.put(light_curve)
        except asyncio.QueueFull:
            self.logger.warning(f"Queue full, dropping light curve {light_curve.star_id}")
    
    async def _processing_loop(self):
        """Main processing loop."""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch
                while len(batch) < self.batch_size and self.is_running:
                    try:
                        # Wait for new data with timeout
                        light_curve = await asyncio.wait_for(
                            self.input_queue.get(), timeout=1.0
                        )
                        batch.append(light_curve)
                    except asyncio.TimeoutError:
                        # Process partial batch if we have data
                        if batch:
                            break
                        continue
                
                # Process batch if we have data
                if batch:
                    await self._process_batch(batch)
                    batch = []
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, light_curves: List[LightCurve]):
        """
        Process a batch of light curves.
        
        Args:
            light_curves: List of light curves to process
        """
        start_time = time.time()
        
        try:
            # Preprocess light curves
            processed_data = []
            valid_indices = []
            
            for i, lc in enumerate(light_curves):
                try:
                    processed = self.preprocessor.process(lc)
                    
                    # Convert to tensor format
                    if hasattr(processed, 'dual_channel_data'):
                        data = processed.dual_channel_data
                    else:
                        # Create dual channel from flux
                        data = np.stack([processed.flux, processed.flux])
                    
                    processed_data.append(data)
                    valid_indices.append(i)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to preprocess {lc.star_id}: {e}")
                    continue
            
            if not processed_data:
                return
            
            # Convert to tensor
            batch_tensor = torch.tensor(
                np.array(processed_data), 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Model inference
            with torch.no_grad():
                predictions = self.model(batch_tensor)
                confidences = predictions.squeeze().cpu().numpy()
            
            # Handle single prediction case
            if confidences.ndim == 0:
                confidences = np.array([confidences])
            
            # Generate alerts for high-confidence detections
            for i, confidence in enumerate(confidences):
                original_idx = valid_indices[i]
                light_curve = light_curves[original_idx]
                
                if confidence >= self.confidence_threshold:
                    alert = DetectionAlert(
                        star_id=light_curve.star_id,
                        confidence=float(confidence),
                        timestamp=datetime.now(),
                        metadata={
                            'processing_time': time.time() - start_time,
                            'batch_size': len(light_curves)
                        }
                    )
                    
                    await self._handle_alert(alert)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['processed_count'] += len(light_curves)
            self.processing_stats['processing_time_total'] += processing_time
            
            self.logger.debug(
                f"Processed batch of {len(light_curves)} light curves in {processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
    
    async def _handle_alert(self, alert: DetectionAlert):
        """
        Handle detection alert.
        
        Args:
            alert: Detection alert to handle
        """
        self.processing_stats['alert_count'] += 1
        
        self.logger.info(
            f"ALERT: {alert.star_id} - Confidence: {alert.confidence:.3f}"
        )
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(alert)
                else:
                    self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.processing_stats['start_time']
        
        stats = self.processing_stats.copy()
        stats.update({
            'elapsed_time': elapsed_time,
            'processing_rate': stats['processed_count'] / elapsed_time if elapsed_time > 0 else 0,
            'average_processing_time': (
                stats['processing_time_total'] / stats['processed_count'] 
                if stats['processed_count'] > 0 else 0
            ),
            'queue_size': self.input_queue.qsize(),
            'is_running': self.is_running
        })
        
        return stats


class AlertManager:
    """
    Manager for handling and storing detection alerts.
    """
    
    def __init__(
        self,
        alert_file: Optional[Path] = None,
        max_alerts: int = 10000
    ):
        """
        Initialize alert manager.
        
        Args:
            alert_file: File to store alerts
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.alert_file = Path(alert_file) if alert_file else None
        self.max_alerts = max_alerts
        
        # Alert storage
        self.alerts = []
        self.alert_lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def add_alert(self, alert: DetectionAlert):
        """
        Add new alert.
        
        Args:
            alert: Detection alert to add
        """
        with self.alert_lock:
            self.alerts.append(alert)
            
            # Limit number of stored alerts
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            # Save to file if specified
            if self.alert_file:
                self._save_alert_to_file(alert)
    
    def _save_alert_to_file(self, alert: DetectionAlert):
        """Save alert to file."""
        try:
            with open(self.alert_file, 'a') as f:
                json.dump(alert.to_dict(), f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save alert to file: {e}")
    
    def get_recent_alerts(self, n: int = 10) -> List[DetectionAlert]:
        """
        Get recent alerts.
        
        Args:
            n: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        with self.alert_lock:
            return self.alerts[-n:] if self.alerts else []
    
    def get_alert_count(self) -> int:
        """Get total number of alerts."""
        with self.alert_lock:
            return len(self.alerts)
    
    def clear_alerts(self):
        """Clear all stored alerts."""
        with self.alert_lock:
            self.alerts.clear()


class StreamingDataSource:
    """
    Mock streaming data source for testing.
    """
    
    def __init__(
        self,
        data_generator: Callable[[], LightCurve],
        rate: float = 1.0
    ):
        """
        Initialize streaming data source.
        
        Args:
            data_generator: Function to generate light curves
            rate: Data generation rate (items per second)
        """
        self.data_generator = data_generator
        self.rate = rate
        self.is_streaming = False
    
    async def stream_data(self) -> AsyncGenerator[LightCurve, None]:
        """
        Stream light curve data.
        
        Yields:
            Light curve data
        """
        self.is_streaming = True
        
        while self.is_streaming:
            try:
                # Generate new light curve
                light_curve = self.data_generator()
                yield light_curve
                
                # Wait based on rate
                await asyncio.sleep(1.0 / self.rate)
                
            except Exception as e:
                logging.error(f"Error generating data: {e}")
                await asyncio.sleep(1.0)
    
    def stop_streaming(self):
        """Stop data streaming."""
        self.is_streaming = False


# Factory functions
def create_real_time_processor(
    model: torch.nn.Module,
    preprocessor: LightCurvePreprocessor,
    **kwargs
) -> RealTimeProcessor:
    """Create real-time processor with default configuration."""
    return RealTimeProcessor(model, preprocessor, **kwargs)


def create_alert_manager(alert_file: Optional[Path] = None) -> AlertManager:
    """Create alert manager."""
    return AlertManager(alert_file=alert_file)