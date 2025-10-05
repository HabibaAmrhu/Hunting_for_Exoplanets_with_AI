"""
Comprehensive Quality Assurance test suite for the exoplanet detection pipeline.
Includes integration tests, performance tests, security tests, and end-to-end validation.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import time
import threading
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import subprocess
import psutil
import logging

# Import project modules
from src.data import ExoplanetDataset, DataAugmentation
from src.models import ExoplanetCNN, TransformerModel, EnsembleModel
from src.training import ExoplanetTrainer, MetricsCalculator
from src.inference import ModelInference
from src.security import SecurityManager, UserRole, Permission
from src.monitoring import ModelMonitor
from src.utils import PerformanceOptimizer
from src.visualization import AdvancedVisualizer


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    def test_dataset_consistency(self, sample_dataset):
        """Test dataset consistency and integrity."""
        dataset = sample_dataset
        
        # Test dataset size consistency
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test data shape consistency
        first_sample = dataset[0]
        data_shape = first_sample[0].shape
        
        for i in range(min(10, len(dataset))):
            sample_data, sample_label = dataset[i]
            assert sample_data.shape == data_shape, f"Inconsistent data shape at index {i}"
            assert isinstance(sample_label, (int, float, torch.Tensor)), f"Invalid label type at index {i}"
    
    def test_data_augmentation_integrity(self):
        """Test data augmentation preserves data integrity."""
        augmenter = DataAugmentation(
            noise_level=0.01,
            time_shift_range=0.1,
            amplitude_scale_range=(0.9, 1.1)
        )
        
        # Create test data
        original_data = np.random.randn(1000)
        
        # Apply augmentations
        noisy_data = augmenter.add_noise(original_data)
        shifted_data = augmenter.time_shift(original_data)
        scaled_data = augmenter.amplitude_scale(original_data)
        
        # Check data integrity
        assert noisy_data.shape == original_data.shape, "Noise augmentation changed data shape"
        assert shifted_data.shape == original_data.shape, "Time shift changed data shape"
        assert scaled_data.shape == original_data.shape, "Amplitude scaling changed data shape"
        
        # Check data is actually different (augmented)
        assert not np.array_equal(noisy_data, original_data), "Noise augmentation had no effect"
        assert not np.array_equal(shifted_data, original_data), "Time shift had no effect"
        assert not np.array_equal(scaled_data, original_data), "Amplitude scaling had no effect"
    
    def test_data_validation(self):
        """Test data validation functions."""
        from src.data.validation import validate_light_curve_data
        
        # Valid data
        valid_data = {
            'time': np.linspace(0, 100, 1000),
            'flux': np.ones(1000) + 0.01 * np.random.randn(1000),
            'flux_err': 0.01 * np.ones(1000)
        }
        
        assert validate_light_curve_data(valid_data), "Valid data should pass validation"
        
        # Invalid data - mismatched lengths
        invalid_data = {
            'time': np.linspace(0, 100, 1000),
            'flux': np.ones(500),  # Wrong length
            'flux_err': 0.01 * np.ones(1000)
        }
        
        assert not validate_light_curve_data(invalid_data), "Invalid data should fail validation"


class TestModelPerformance:
    """Test model performance and accuracy."""
    
    @pytest.mark.parametrize("model_class", [ExoplanetCNN, TransformerModel])
    def test_model_accuracy_threshold(self, model_class, sample_data_loader):
        """Test that models meet minimum accuracy thresholds."""
        model = model_class(input_length=1000)
        
        # Quick training for testing
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        model.train()
        for epoch in range(5):  # Quick training
            for batch_data, batch_labels in sample_data_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                loss.backward()
                optimizer.step()
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in sample_data_loader:
                outputs = model(batch_data)
                predicted = (outputs.squeeze() > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels.float()).sum().item()
        
        accuracy = correct / total
        assert accuracy > 0.5, f"Model accuracy {accuracy:.3f} below minimum threshold"
    
    def test_ensemble_performance(self, sample_data_loader):
        """Test ensemble model performance."""
        # Create individual models
        models = [
            ExoplanetCNN(input_length=1000),
            TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=2)
        ]
        
        # Create ensemble
        ensemble = EnsembleModel(models, weights=[0.6, 0.4])
        
        # Test ensemble predictions
        ensemble.eval()
        with torch.no_grad():
            for batch_data, batch_labels in sample_data_loader:
                outputs = ensemble(batch_data)
                assert outputs.shape[0] == batch_data.shape[0], "Ensemble output shape mismatch"
                assert torch.all(outputs >= 0) and torch.all(outputs <= 1), "Ensemble outputs not in [0,1]"
                break  # Test one batch
    
    def test_model_inference_speed(self):
        """Test model inference speed requirements."""
        model = ExoplanetCNN(input_length=1000)
        model.eval()
        
        # Prepare test data
        test_data = torch.randn(100, 1, 1000)
        
        # Warmup
        with torch.no_grad():
            _ = model(test_data[:10])
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            predictions = model(test_data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        samples_per_second = len(test_data) / inference_time
        
        # Assert minimum performance (adjust threshold as needed)
        assert samples_per_second > 50, f"Inference too slow: {samples_per_second:.1f} samples/sec"


class TestSecurityCompliance:
    """Test security features and compliance."""
    
    def test_authentication_system(self):
        """Test authentication system security."""
        security = SecurityManager()
        
        # Test user creation with weak password
        success, message = security.create_user(
            "testuser", "test@example.com", "weak", UserRole.VIEWER
        )
        assert not success, "Weak password should be rejected"
        
        # Test user creation with strong password
        success, message = security.create_user(
            "testuser", "test@example.com", "StrongPass123!", UserRole.VIEWER
        )
        assert success, "Strong password should be accepted"
        
        # Test authentication with wrong password
        success, user_id, error = security.authenticate_user("testuser", "wrongpass")
        assert not success, "Wrong password should fail authentication"
        
        # Test authentication with correct password
        success, user_id, error = security.authenticate_user("testuser", "StrongPass123!")
        assert success, "Correct password should succeed authentication"
        
        # Test token generation and verification
        token = security.generate_token(user_id)
        is_valid, payload = security.verify_token(token)
        assert is_valid, "Generated token should be valid"
        assert payload['user_id'] == user_id, "Token payload should contain correct user_id"
    
    def test_permission_system(self):
        """Test role-based permission system."""
        security = SecurityManager()
        
        # Create users with different roles
        security.create_user("admin", "admin@example.com", "AdminPass123!", UserRole.ADMIN)
        security.create_user("viewer", "viewer@example.com", "ViewerPass123!", UserRole.VIEWER)
        
        # Get user IDs
        _, admin_id, _ = security.authenticate_user("admin", "AdminPass123!")
        _, viewer_id, _ = security.authenticate_user("viewer", "ViewerPass123!")
        
        # Test admin permissions
        assert security.has_permission(admin_id, Permission.MANAGE_USERS), "Admin should have manage users permission"
        assert security.has_permission(admin_id, Permission.TRAIN_MODELS), "Admin should have train models permission"
        
        # Test viewer permissions
        assert security.has_permission(viewer_id, Permission.VIEW_METRICS), "Viewer should have view metrics permission"
        assert not security.has_permission(viewer_id, Permission.MANAGE_USERS), "Viewer should not have manage users permission"
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        from src.security.validation import validate_input_data, sanitize_file_path
        
        # Test valid input
        valid_data = np.random.randn(1000)
        assert validate_input_data(valid_data), "Valid data should pass validation"
        
        # Test invalid input - wrong type
        with pytest.raises(TypeError):
            validate_input_data("not an array")
        
        # Test invalid input - NaN values
        invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError):
            validate_input_data(invalid_data)
        
        # Test path sanitization
        safe_path = sanitize_file_path("data/test.csv")
        assert "data" in str(safe_path), "Safe path should be preserved"
        
        # Test path traversal prevention
        with pytest.raises(ValueError):
            sanitize_file_path("../../../etc/passwd")


class TestSystemIntegration:
    """Test system integration and end-to-end workflows."""
    
    def test_training_pipeline_integration(self, sample_dataset):
        """Test complete training pipeline integration."""
        # Create model and data loader
        model = ExoplanetCNN(input_length=1000)
        train_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=16, shuffle=False)
        
        # Create trainer
        trainer = ExoplanetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device=torch.device('cpu'),
            experiment_name="test_integration"
        )
        
        # Run training
        history = trainer.train(epochs=2, verbose=False)
        
        # Validate training results
        assert 'train_loss' in history, "Training history should contain train_loss"
        assert 'val_loss' in history, "Training history should contain val_loss"
        assert len(history['train_loss']) == 2, "Should have loss for each epoch"
    
    def test_inference_pipeline_integration(self, sample_dataset):
        """Test complete inference pipeline integration."""
        # Create and save a model
        model = ExoplanetCNN(input_length=1000)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
            torch.save(model.state_dict(), model_path)
        
        try:
            # Create inference engine
            inference = ModelInference(model_path, device='cpu')
            
            # Test single prediction
            sample_data = torch.randn(1, 1, 1000)
            prediction = inference.predict(sample_data)
            
            assert prediction.shape == (1,), "Single prediction should have shape (1,)"
            assert 0 <= prediction[0] <= 1, "Prediction should be between 0 and 1"
            
            # Test batch prediction
            batch_data = torch.randn(10, 1, 1000)
            batch_predictions = inference.predict(batch_data)
            
            assert batch_predictions.shape == (10,), "Batch predictions should have shape (10,)"
            assert np.all((batch_predictions >= 0) & (batch_predictions <= 1)), "All predictions should be between 0 and 1"
            
        finally:
            # Cleanup
            Path(model_path).unlink()
    
    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        # Create mock model and reference data
        model = ExoplanetCNN(input_length=1000)
        reference_data = torch.randn(100, 1, 1000)
        
        # Create monitor
        monitor = ModelMonitor(
            model=model,
            reference_data=reference_data,
            alert_thresholds={'accuracy': 0.8, 'drift_score': 0.1}
        )
        
        # Test performance monitoring
        predictions = torch.rand(50)
        targets = torch.randint(0, 2, (50,)).float()
        
        performance_report = monitor.check_performance(predictions.numpy(), targets.numpy())
        
        assert 'accuracy' in performance_report, "Performance report should contain accuracy"
        assert 'alerts' in performance_report, "Performance report should contain alerts"
        
        # Test drift detection
        new_data = torch.randn(50, 1, 1000)  # Similar to reference
        drift_report = monitor.detect_drift(new_data.numpy())
        
        assert 'drift_score' in drift_report, "Drift report should contain drift_score"
        assert 'is_drift' in drift_report, "Drift report should contain is_drift flag"


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_memory_management(self):
        """Test memory management utilities."""
        optimizer = PerformanceOptimizer()
        
        # Test memory context manager
        initial_memory = optimizer._get_memory_usage()
        
        with optimizer.memory_management():
            # Allocate some memory
            large_tensor = torch.randn(1000, 1000)
            del large_tensor
        
        final_memory = optimizer._get_memory_usage()
        
        # Memory should be managed (not necessarily lower due to Python GC behavior)
        assert len(optimizer.memory_stats) > 0, "Memory statistics should be recorded"
    
    def test_batch_processing_optimization(self):
        """Test optimized batch processing."""
        model = ExoplanetCNN(input_length=1000)
        optimizer = PerformanceOptimizer()
        
        # Create batch processor
        from src.utils.performance import BatchProcessor
        processor = BatchProcessor(model, torch.device('cpu'), batch_size=32)
        
        # Create test dataset
        test_data = torch.randn(100, 1, 1000)
        test_dataset = torch.utils.data.TensorDataset(test_data, torch.zeros(100))
        
        # Process dataset
        predictions, uncertainties = processor.process_dataset(test_dataset)
        
        assert predictions.shape == (100,), "Predictions should match dataset size"
        assert uncertainties.shape == (100,), "Uncertainties should match dataset size"
    
    def test_model_optimization(self):
        """Test model optimization for inference."""
        model = ExoplanetCNN(input_length=1000)
        optimizer = PerformanceOptimizer()
        
        # Optimize model
        optimized_model = optimizer.optimize_model_for_inference(model)
        
        # Test that model is in eval mode
        assert not optimized_model.training, "Optimized model should be in eval mode"
        
        # Test that gradients are disabled
        for param in optimized_model.parameters():
            assert not param.requires_grad, "Optimized model parameters should not require gradients"


class TestVisualizationQuality:
    """Test visualization and reporting quality."""
    
    def test_visualization_generation(self):
        """Test visualization generation without errors."""
        visualizer = AdvancedVisualizer()
        
        # Test performance comparison plot
        results = {
            'Model A': {'f1_score': 0.85, 'precision': 0.82, 'recall': 0.88, 'roc_auc': 0.90},
            'Model B': {'f1_score': 0.78, 'precision': 0.80, 'recall': 0.76, 'roc_auc': 0.85}
        }
        
        fig = visualizer.plot_performance_comparison(results)
        assert fig is not None, "Performance comparison plot should be generated"
        
        # Test learning curves plot
        training_history = {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
            'val_loss': [0.7, 0.5, 0.4, 0.35, 0.25],
            'train_metrics': [
                {'f1_score': 0.6, 'precision': 0.65, 'recall': 0.55},
                {'f1_score': 0.7, 'precision': 0.72, 'recall': 0.68},
                {'f1_score': 0.8, 'precision': 0.82, 'recall': 0.78},
                {'f1_score': 0.85, 'precision': 0.87, 'recall': 0.83},
                {'f1_score': 0.88, 'precision': 0.90, 'recall': 0.86}
            ],
            'val_metrics': [
                {'f1_score': 0.58, 'precision': 0.62, 'recall': 0.54},
                {'f1_score': 0.68, 'precision': 0.70, 'recall': 0.66},
                {'f1_score': 0.78, 'precision': 0.80, 'recall': 0.76},
                {'f1_score': 0.82, 'precision': 0.84, 'recall': 0.80},
                {'f1_score': 0.85, 'precision': 0.87, 'recall': 0.83}
            ]
        }
        
        fig = visualizer.plot_learning_curves(training_history)
        assert fig is not None, "Learning curves plot should be generated"
    
    def test_report_generation(self):
        """Test automated report generation."""
        # Test metrics calculation
        calculator = MetricsCalculator()
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.6, 0.9, 0.8, 0.1, 0.3])
        
        metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Validate required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics, f"Metric {metric} should be in results"
            assert 0 <= metrics[metric] <= 1, f"Metric {metric} should be between 0 and 1"


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    def test_graceful_error_handling(self):
        """Test that system handles errors gracefully."""
        # Test model loading with invalid path
        with pytest.raises(FileNotFoundError):
            ModelInference("nonexistent_model.pth")
        
        # Test dataset with invalid data
        with pytest.raises((ValueError, TypeError)):
            ExoplanetDataset("nonexistent_directory")
    
    def test_input_validation_errors(self):
        """Test input validation error handling."""
        model = ExoplanetCNN(input_length=1000)
        
        # Test with wrong input shape
        with pytest.raises((RuntimeError, ValueError)):
            wrong_shape_input = torch.randn(10, 2, 500)  # Wrong shape
            model(wrong_shape_input)
    
    def test_memory_error_recovery(self):
        """Test recovery from memory errors."""
        optimizer = PerformanceOptimizer()
        
        # Test memory management under stress
        try:
            with optimizer.memory_management():
                # Try to allocate large amount of memory
                large_tensors = []
                for i in range(10):
                    large_tensors.append(torch.randn(1000, 1000))
        except RuntimeError as e:
            # Should handle memory errors gracefully
            assert "memory" in str(e).lower() or "cuda" in str(e).lower()


class TestConcurrencyAndThreadSafety:
    """Test concurrent operations and thread safety."""
    
    def test_concurrent_predictions(self):
        """Test concurrent model predictions."""
        model = ExoplanetCNN(input_length=1000)
        model.eval()
        
        results = []
        errors = []
        
        def make_prediction(thread_id):
            try:
                data = torch.randn(1, 1, 1000)
                with torch.no_grad():
                    prediction = model(data)
                results.append((thread_id, prediction.item()))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_prediction, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent predictions failed: {errors}"
        assert len(results) == 5, "All threads should complete successfully"
    
    def test_thread_safe_monitoring(self):
        """Test thread-safe monitoring operations."""
        model = ExoplanetCNN(input_length=1000)
        reference_data = torch.randn(100, 1, 1000)
        monitor = ModelMonitor(model, reference_data)
        
        results = []
        errors = []
        
        def monitor_performance(thread_id):
            try:
                predictions = np.random.rand(50)
                targets = np.random.randint(0, 2, 50).astype(float)
                report = monitor.check_performance(predictions, targets)
                results.append((thread_id, report))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple monitoring threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=monitor_performance, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent monitoring failed: {errors}"
        assert len(results) == 3, "All monitoring threads should complete"


class TestResourceUsage:
    """Test resource usage and limits."""
    
    def test_memory_usage_limits(self):
        """Test that memory usage stays within reasonable limits."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create and use model
        model = ExoplanetCNN(input_length=1000)
        data = torch.randn(100, 1, 1000)
        
        with torch.no_grad():
            predictions = model(data)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f} MB"
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency."""
        model = ExoplanetCNN(input_length=1000)
        model.eval()
        
        # Measure CPU time for predictions
        data = torch.randn(100, 1, 1000)
        
        start_time = time.process_time()
        with torch.no_grad():
            predictions = model(data)
        end_time = time.process_time()
        
        cpu_time = end_time - start_time
        samples_per_cpu_second = len(data) / cpu_time
        
        # Should process at least 10 samples per CPU second
        assert samples_per_cpu_second > 10, f"CPU efficiency too low: {samples_per_cpu_second:.1f} samples/CPU-sec"


# Fixtures for testing
@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    # Create synthetic data
    data = []
    labels = []
    
    for i in range(100):
        # Generate synthetic light curve
        time_series = np.random.randn(1000)
        if i % 2 == 0:  # Half with planets
            # Add transit signal
            transit_start = np.random.randint(200, 800)
            transit_duration = np.random.randint(10, 50)
            time_series[transit_start:transit_start + transit_duration] *= 0.99
            label = 1
        else:
            label = 0
        
        data.append(torch.FloatTensor(time_series).unsqueeze(0))
        labels.append(label)
    
    return torch.utils.data.TensorDataset(torch.stack(data), torch.tensor(labels))


@pytest.fixture
def sample_data_loader(sample_dataset):
    """Create a sample data loader for testing."""
    return torch.utils.data.DataLoader(sample_dataset, batch_size=16, shuffle=True)


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_training_speed_benchmark(self, benchmark, sample_data_loader):
        """Benchmark training speed."""
        model = ExoplanetCNN(input_length=1000)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.BCELoss()
        
        def train_one_epoch():
            model.train()
            for batch_data, batch_labels in sample_data_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                loss.backward()
                optimizer.step()
        
        # Benchmark training
        result = benchmark(train_one_epoch)
        
        # Training should complete within reasonable time
        assert result < 10.0, f"Training epoch took {result:.2f}s, too slow"
    
    def test_inference_speed_benchmark(self, benchmark):
        """Benchmark inference speed."""
        model = ExoplanetCNN(input_length=1000)
        model.eval()
        data = torch.randn(100, 1, 1000)
        
        def run_inference():
            with torch.no_grad():
                return model(data)
        
        # Benchmark inference
        result = benchmark(run_inference)
        
        # Inference should be fast
        assert result < 1.0, f"Inference took {result:.2f}s, too slow"


if __name__ == "__main__":
    # Run comprehensive test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ])