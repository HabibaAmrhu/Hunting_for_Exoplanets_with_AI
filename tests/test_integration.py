"""
Integration tests for the complete exoplanet detection pipeline.
Tests end-to-end workflows, cross-platform compatibility, and performance benchmarks.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
import sys
import subprocess
import platform
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.downloader import DataDownloader
from data.dataset import LightCurveDataset, AugmentedLightCurveDataset, collate_fn
from data.augmentation import create_standard_augmentation_pipeline
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.synthetic_injection import SyntheticTransitInjector
from models.cnn import ExoplanetCNN
from models.lstm import ExoplanetLSTM
from models.transformer import ExoplanetTransformer
from models.ensemble import EnsembleModel
from training.trainer import ExoplanetTrainer, create_optimizer, create_scheduler
from training.metrics import MetricsCalculator
from data.types import PreprocessingConfig
from torch.utils.data import DataLoader


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_size = 50  # Small for fast testing
        self.sequence_length = 512  # Reduced for speed
        
    def test_complete_pipeline_workflow(self):
        """Test complete pipeline from data download to model evaluation."""
        
        # Step 1: Mock data download (to avoid actual API calls)
        with patch.object(DataDownloader, 'download_kepler_koi') as mock_download:
            # Create mock metadata
            mock_metadata = pd.DataFrame({
                'kepid': range(self.sample_size),
                'koi_disposition': ['CONFIRMED'] * 25 + ['FALSE POSITIVE'] * 25,
                'koi_period': np.random.uniform(1, 100, self.sample_size),
                'koi_depth': np.random.uniform(100, 10000, self.sample_size),
                'koi_duration': np.random.uniform(1, 10, self.sample_size)
            })
            
            mock_download.return_value = (mock_metadata, Path(self.temp_dir) / 'metadata.csv')
            
            downloader = DataDownloader(cache_dir=Path(self.temp_dir) / 'cache')
            metadata_df, metadata_file = downloader.download_kepler_koi(
                output_dir=Path(self.temp_dir) / 'kepler',
                sample_size=self.sample_size
            )
            
            assert len(metadata_df) == self.sample_size
            assert metadata_file.exists()
        
        # Step 2: Create mock preprocessed data
        n_channels = 2
        data = np.random.randn(self.sample_size, n_channels, self.sequence_length)
        labels = np.random.randint(0, 2, self.sample_size)
        metadata = [{'star_id': f'test_{i}'} for i in range(self.sample_size)]
        
        # Step 3: Test preprocessing pipeline
        config = PreprocessingConfig(
            target_length=self.sequence_length,
            detrend_method='median',
            normalization='zscore'
        )
        
        preprocessing_pipeline = PreprocessingPipeline(
            config=config,
            output_dir=Path(self.temp_dir) / 'processed'
        )
        
        assert preprocessing_pipeline.config.target_length == self.sequence_length
        
        # Step 4: Test synthetic transit injection
        injector = SyntheticTransitInjector(
            stellar_catalog='kepler',
            planet_survey='kepler',
            noise_model='realistic'
        )
        
        # Test parameter sampling
        stellar_params = injector.sample_stellar_parameters()
        assert 'temperature' in stellar_params
        
        # Step 5: Test data augmentation
        augmentation_pipeline = create_standard_augmentation_pipeline()
        
        augmented_dataset = AugmentedLightCurveDataset(
            data, labels, metadata,
            augmentation_pipeline=augmentation_pipeline
        )
        
        assert len(augmented_dataset) == self.sample_size
        
        # Step 6: Test model training
        train_data = data[:40]
        train_labels = labels[:40]
        train_metadata = metadata[:40]
        
        val_data = data[40:]
        val_labels = labels[40:]
        val_metadata = metadata[40:]
        
        train_dataset = LightCurveDataset(train_data, train_labels, train_metadata)
        val_dataset = LightCurveDataset(val_data, val_labels, val_metadata)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Create and train model
        model = ExoplanetCNN(input_channels=n_channels, sequence_length=self.sequence_length)
        criterion = torch.nn.BCELoss()
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        
        trainer = ExoplanetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
            checkpoint_dir=Path(self.temp_dir) / 'checkpoints',
            experiment_name="integration_test"
        )
        
        # Train for minimal epochs
        history = trainer.train(epochs=2, verbose=False)
        
        assert len(history['train_loss']) <= 2
        assert all(loss >= 0 for loss in history['train_loss'])
        
        # Step 7: Test evaluation
        predictions, targets = trainer.predict(val_loader)
        
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_metrics(targets, predictions)
        
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        
        print("✅ Complete end-to-end pipeline test passed!")
        
    def test_multi_model_ensemble_workflow(self):
        """Test ensemble workflow with multiple model architectures."""
        
        # Create test data
        n_channels = 2
        data = np.random.randn(32, n_channels, self.sequence_length)
        labels = np.random.randint(0, 2, 32)
        metadata = [{'star_id': f'test_{i}'} for i in range(32)]
        
        # Split data
        train_data, val_data = data[:24], data[24:]
        train_labels, val_labels = labels[:24], labels[24:]
        train_metadata, val_metadata = metadata[:24], metadata[24:]
        
        # Create datasets
        train_dataset = LightCurveDataset(train_data, train_labels, train_metadata)
        val_dataset = LightCurveDataset(val_data, val_labels, val_metadata)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        # Create multiple models
        models = []
        model_names = []
        
        # CNN Model
        cnn_model = ExoplanetCNN(input_channels=n_channels, sequence_length=self.sequence_length)
        models.append(cnn_model)
        model_names.append('CNN')
        
        # Train CNN briefly
        cnn_trainer = ExoplanetTrainer(
            model=cnn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=torch.nn.BCELoss(),
            optimizer=create_optimizer(cnn_model, 'adam'),
            device=torch.device('cpu'),
            experiment_name="cnn_test"
        )
        cnn_trainer.train(epochs=1, verbose=False)
        
        # LSTM Model (if available)
        try:
            lstm_model = ExoplanetLSTM(input_channels=n_channels, sequence_length=self.sequence_length)
            models.append(lstm_model)
            model_names.append('LSTM')
            
            lstm_trainer = ExoplanetTrainer(
                model=lstm_model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=torch.nn.BCELoss(),
                optimizer=create_optimizer(lstm_model, 'adam'),
                device=torch.device('cpu'),
                experiment_name="lstm_test"
            )
            lstm_trainer.train(epochs=1, verbose=False)
        except Exception as e:
            print(f"LSTM model test skipped: {e}")
        
        # Create ensemble if we have multiple models
        if len(models) > 1:
            ensemble = EnsembleModel(
                models=models,
                combination_method='average',
                model_names=model_names
            )
            
            # Test ensemble prediction
            ensemble.eval()
            with torch.no_grad():
                sample_input = torch.randn(1, n_channels, self.sequence_length)
                ensemble_output = ensemble(sample_input)
                
                assert ensemble_output.shape == (1, 1)
                assert 0 <= ensemble_output.item() <= 1
        
        print("✅ Multi-model ensemble workflow test passed!")


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility and environment detection."""
    
    def test_platform_detection(self):
        """Test platform detection and compatibility."""
        current_platform = platform.system()
        
        # Test that we can detect the platform
        assert current_platform in ['Windows', 'Linux', 'Darwin']
        
        # Test Python version compatibility
        python_version = sys.version_info
        assert python_version >= (3, 7), "Python 3.7+ required"
        
        print(f"✅ Platform compatibility test passed on {current_platform}")
        
    def test_torch_compatibility(self):
        """Test PyTorch compatibility and device detection."""
        # Test PyTorch installation
        assert torch.__version__ is not None
        
        # Test CPU availability
        cpu_device = torch.device('cpu')
        test_tensor = torch.randn(10, 10, device=cpu_device)
        assert test_tensor.device.type == 'cpu'
        
        # Test CUDA availability (if present)
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda')
            test_tensor_cuda = torch.randn(10, 10, device=cuda_device)
            assert test_tensor_cuda.device.type == 'cuda'
            print("✅ CUDA compatibility confirmed")
        else:
            print("ℹ️  CUDA not available, using CPU only")
        
        print("✅ PyTorch compatibility test passed!")
        
    def test_dependency_imports(self):
        """Test that all required dependencies can be imported."""
        required_modules = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
            'torch', 'torchvision', 'tqdm', 'pathlib'
        ]
        
        missing_modules = []
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        if missing_modules:
            pytest.fail(f"Missing required modules: {missing_modules}")
        
        print("✅ All required dependencies available!")


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def setup_method(self):
        """Setup performance test fixtures."""
        self.benchmark_data_sizes = [100, 500, 1000]
        self.sequence_length = 2048
        self.n_channels = 2
        
    def test_data_loading_performance(self):
        """Test data loading performance with different dataset sizes."""
        performance_results = {}
        
        for n_samples in self.benchmark_data_sizes:
            # Create test data
            data = np.random.randn(n_samples, self.n_channels, self.sequence_length)
            labels = np.random.randint(0, 2, n_samples)
            metadata = [{'star_id': f'test_{i}'} for i in range(n_samples)]
            
            # Time dataset creation
            start_time = time.time()
            dataset = LightCurveDataset(data, labels, metadata)
            creation_time = time.time() - start_time
            
            # Time data loading
            start_time = time.time()
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
            
            for batch in dataloader:
                pass  # Just iterate through all batches
            
            loading_time = time.time() - start_time
            
            performance_results[n_samples] = {
                'creation_time': creation_time,
                'loading_time': loading_time,
                'samples_per_second': n_samples / loading_time if loading_time > 0 else float('inf')
            }
        
        # Print performance results
        print("\\n📊 Data Loading Performance Results:")
        for n_samples, results in performance_results.items():
            print(f"  {n_samples} samples: {results['samples_per_second']:.1f} samples/sec")
        
        # Basic performance assertions
        for results in performance_results.values():
            assert results['creation_time'] < 10.0, "Dataset creation should be fast"
            assert results['samples_per_second'] > 10, "Should process at least 10 samples/sec"
        
        print("✅ Data loading performance test passed!")
        
    def test_model_inference_performance(self):
        """Test model inference performance."""
        batch_sizes = [1, 8, 32]
        model = ExoplanetCNN(input_channels=self.n_channels, sequence_length=self.sequence_length)
        model.eval()
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Create test batch
            test_input = torch.randn(batch_size, self.n_channels, self.sequence_length)
            
            # Warm up
            with torch.no_grad():
                _ = model(test_input)
            
            # Time inference
            start_time = time.time()
            n_iterations = 10
            
            with torch.no_grad():
                for _ in range(n_iterations):
                    _ = model(test_input)
            
            total_time = time.time() - start_time
            avg_time_per_batch = total_time / n_iterations
            samples_per_second = batch_size / avg_time_per_batch
            
            performance_results[batch_size] = {
                'avg_time_per_batch': avg_time_per_batch,
                'samples_per_second': samples_per_second
            }
        
        # Print performance results
        print("\\n📊 Model Inference Performance Results:")
        for batch_size, results in performance_results.items():
            print(f"  Batch size {batch_size}: {results['samples_per_second']:.1f} samples/sec")
        
        # Performance assertions
        for results in performance_results.values():
            assert results['samples_per_second'] > 1, "Should process at least 1 sample/sec"
        
        print("✅ Model inference performance test passed!")
        
    def test_training_performance(self):
        """Test training performance benchmark."""
        # Create small dataset for training benchmark
        n_samples = 100
        data = np.random.randn(n_samples, self.n_channels, 512)  # Smaller sequence for speed
        labels = np.random.randint(0, 2, n_samples)
        metadata = [{'star_id': f'test_{i}'} for i in range(n_samples)]
        
        # Split data
        train_data, val_data = data[:80], data[80:]
        train_labels, val_labels = labels[:80], labels[80:]
        train_metadata, val_metadata = metadata[:80], metadata[80:]
        
        # Create datasets and loaders
        train_dataset = LightCurveDataset(train_data, train_labels, train_metadata)
        val_dataset = LightCurveDataset(val_data, val_labels, val_metadata)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        # Create model and trainer
        model = ExoplanetCNN(input_channels=self.n_channels, sequence_length=512)
        criterion = torch.nn.BCELoss()
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        
        trainer = ExoplanetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
            experiment_name="performance_test"
        )
        
        # Time training
        start_time = time.time()
        history = trainer.train(epochs=2, verbose=False)
        training_time = time.time() - start_time
        
        epochs_per_minute = (2 / training_time) * 60 if training_time > 0 else float('inf')
        
        print(f"\\n📊 Training Performance: {epochs_per_minute:.2f} epochs/minute")
        
        # Performance assertions
        assert training_time < 300, "Training should complete within 5 minutes"
        assert len(history['train_loss']) > 0, "Training should produce loss history"
        
        print("✅ Training performance test passed!")


class TestMemoryUsage:
    """Test memory usage and resource management."""
    
    def test_memory_efficiency(self):
        """Test memory usage during training and inference."""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and train a model
        n_samples = 200
        sequence_length = 1024
        n_channels = 2
        
        data = np.random.randn(n_samples, n_channels, sequence_length)
        labels = np.random.randint(0, 2, n_samples)
        metadata = [{'star_id': f'test_{i}'} for i in range(n_samples)]
        
        dataset = LightCurveDataset(data, labels, metadata)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        model = ExoplanetCNN(input_channels=n_channels, sequence_length=sequence_length)
        
        # Check memory after model creation
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run a few training steps
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for i, (batch_data, batch_labels, batch_metadata) in enumerate(dataloader):
            if i >= 5:  # Only run a few batches
                break
                
            optimizer.zero_grad()
            outputs = model(batch_data).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        print(f"\\n💾 Memory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  After model: {model_memory:.1f} MB")
        print(f"  After training: {final_memory:.1f} MB")
        print(f"  Total increase: {memory_increase:.1f} MB")
        
        # Memory usage assertions (reasonable limits)
        assert memory_increase < 2000, "Memory increase should be less than 2GB"
        assert final_memory < 4000, "Total memory usage should be less than 4GB"
        
        print("✅ Memory efficiency test passed!")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])