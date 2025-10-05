"""
Comprehensive unit tests for training and evaluation components.
Tests trainer, metrics, and evaluation frameworks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.trainer import ExoplanetTrainer, create_optimizer, create_scheduler
from training.metrics import MetricsCalculator
from models.cnn import ExoplanetCNN
from data.dataset import LightCurveDataset, collate_fn
from torch.utils.data import DataLoader


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.metrics_calc = MetricsCalculator(threshold=0.5)
        
        # Create test predictions and targets
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 100)
        self.y_pred_proba = np.random.random(100)
        
    def test_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        metrics = self.metrics_calc.calculate_metrics(self.y_true, self.y_pred_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc',
            'specificity', 'npv', 'balanced_accuracy', 'mcc',
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
        
    def test_optimal_threshold_finding(self):
        """Test optimal threshold finding."""
        optimal_threshold, optimal_f1 = self.metrics_calc.find_optimal_threshold(
            self.y_true, self.y_pred_proba, metric='f1'
        )
        
        assert 0 <= optimal_threshold <= 1
        assert 0 <= optimal_f1 <= 1
        
        # Test with different metrics
        opt_thresh_prec, opt_prec = self.metrics_calc.find_optimal_threshold(
            self.y_true, self.y_pred_proba, metric='precision'
        )
        
        assert 0 <= opt_thresh_prec <= 1
        assert 0 <= opt_prec <= 1
        
    def test_multiple_thresholds_analysis(self):
        """Test metrics calculation at multiple thresholds."""
        results = self.metrics_calc.calculate_metrics_at_multiple_thresholds(
            self.y_true, self.y_pred_proba
        )
        
        assert 'thresholds' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        
        # Check array lengths match
        n_thresholds = len(results['thresholds'])
        assert len(results['precision']) == n_thresholds
        assert len(results['recall']) == n_thresholds
        assert len(results['f1_score']) == n_thresholds
        
    def test_edge_cases(self):
        """Test edge cases for metrics calculation."""
        # All positive predictions
        y_true_all_pos = np.ones(50)
        y_pred_all_pos = np.ones(50)
        
        metrics_all_pos = self.metrics_calc.calculate_metrics(y_true_all_pos, y_pred_all_pos)
        assert metrics_all_pos['accuracy'] == 1.0
        assert metrics_all_pos['precision'] == 1.0
        assert metrics_all_pos['recall'] == 1.0
        
        # All negative predictions
        y_true_all_neg = np.zeros(50)
        y_pred_all_neg = np.zeros(50)
        
        metrics_all_neg = self.metrics_calc.calculate_metrics(y_true_all_neg, y_pred_all_neg)
        assert metrics_all_neg['accuracy'] == 1.0
        
    def test_comprehensive_report(self):
        """Test comprehensive evaluation report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = self.metrics_calc.create_comprehensive_report(
                self.y_true, self.y_pred_proba,
                model_name="TestModel",
                save_dir=temp_dir
            )
            
            assert 'model_name' in report
            assert 'default_metrics' in report
            assert 'optimal_f1_metrics' in report
            assert 'optimal_thresholds' in report
            assert 'data_summary' in report
            
            # Check that plots were saved
            plot_files = list(Path(temp_dir).glob("*.png"))
            assert len(plot_files) > 0


class TestTrainerComponents:
    """Test suite for trainer components."""
    
    def test_optimizer_creation(self):
        """Test optimizer factory function."""
        # Create a simple model for testing
        model = nn.Linear(10, 1)
        
        # Test Adam optimizer
        adam_opt = create_optimizer(model, 'adam', learning_rate=0.001)
        assert isinstance(adam_opt, torch.optim.Adam)
        assert adam_opt.param_groups[0]['lr'] == 0.001
        
        # Test AdamW optimizer
        adamw_opt = create_optimizer(model, 'adamw', learning_rate=0.002, weight_decay=0.01)
        assert isinstance(adamw_opt, torch.optim.AdamW)
        assert adamw_opt.param_groups[0]['lr'] == 0.002
        assert adamw_opt.param_groups[0]['weight_decay'] == 0.01
        
        # Test SGD optimizer
        sgd_opt = create_optimizer(model, 'sgd', learning_rate=0.01, momentum=0.9)
        assert isinstance(sgd_opt, torch.optim.SGD)
        assert sgd_opt.param_groups[0]['lr'] == 0.01
        assert sgd_opt.param_groups[0]['momentum'] == 0.9
        
        # Test invalid optimizer
        with pytest.raises(ValueError):
            create_optimizer(model, 'invalid_optimizer')
            
    def test_scheduler_creation(self):
        """Test scheduler factory function."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test cosine annealing scheduler
        cosine_sched = create_scheduler(optimizer, 'cosine', T_max=50)
        assert isinstance(cosine_sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test plateau scheduler
        plateau_sched = create_scheduler(optimizer, 'plateau', patience=5)
        assert isinstance(plateau_sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        # Test step scheduler
        step_sched = create_scheduler(optimizer, 'step', step_size=10)
        assert isinstance(step_sched, torch.optim.lr_scheduler.StepLR)
        
        # Test no scheduler
        no_sched = create_scheduler(optimizer, 'none')
        assert no_sched is None
        
        # Test invalid scheduler
        with pytest.raises(ValueError):
            create_scheduler(optimizer, 'invalid_scheduler')


class TestExoplanetTrainer:
    """Test suite for ExoplanetTrainer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock data
        self.n_samples = 20
        self.sequence_length = 512  # Smaller for faster testing
        self.n_channels = 2
        
        data = np.random.randn(self.n_samples, self.n_channels, self.sequence_length)
        labels = np.random.randint(0, 2, self.n_samples)
        metadata = [{'star_id': f'test_{i}'} for i in range(self.n_samples)]
        
        # Split into train/val
        train_data = data[:16]
        train_labels = labels[:16]
        train_metadata = metadata[:16]
        
        val_data = data[16:]
        val_labels = labels[16:]
        val_metadata = metadata[16:]
        
        # Create datasets
        train_dataset = LightCurveDataset(train_data, train_labels, train_metadata)
        val_dataset = LightCurveDataset(val_data, val_labels, val_metadata)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
        )
        
        # Create model and training components
        self.model = ExoplanetCNN(
            input_channels=self.n_channels,
            sequence_length=self.sequence_length
        )
        self.criterion = nn.BCELoss()
        self.optimizer = create_optimizer(self.model, 'adam', learning_rate=0.001)
        self.scheduler = create_scheduler(self.optimizer, 'cosine', T_max=5)
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ExoplanetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=torch.device('cpu'),
            experiment_name="test_experiment"
        )
        
        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.experiment_name == "test_experiment"
        
    def test_single_epoch_training(self):
        """Test single epoch training."""
        trainer = ExoplanetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=torch.device('cpu'),
            experiment_name="test_single_epoch"
        )
        
        # Test training epoch
        train_metrics = trainer.train_epoch()
        
        assert 'loss' in train_metrics
        assert 'f1_score' in train_metrics
        assert 'accuracy' in train_metrics
        assert train_metrics['loss'] >= 0
        
        # Test validation epoch
        val_metrics = trainer.validate_epoch()
        
        assert 'loss' in val_metrics
        assert 'f1_score' in val_metrics
        assert 'accuracy' in val_metrics
        assert val_metrics['loss'] >= 0
        
    def test_full_training_loop(self):
        """Test full training loop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ExoplanetTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=torch.device('cpu'),
                checkpoint_dir=temp_dir,
                experiment_name="test_full_training"
            )
            
            # Train for a few epochs
            history = trainer.train(epochs=3, patience=10, verbose=False)
            
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert 'train_metrics' in history
            assert 'val_metrics' in history
            assert len(history['train_loss']) <= 3  # Should be <= epochs
            
    def test_checkpointing(self):
        """Test model checkpointing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ExoplanetTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=torch.device('cpu'),
                checkpoint_dir=temp_dir,
                experiment_name="test_checkpoint"
            )
            
            # Save a checkpoint
            trainer.save_checkpoint(epoch=0, is_best=True)
            
            # Check that checkpoint file exists
            checkpoint_files = list(Path(temp_dir).glob("*.pt"))
            assert len(checkpoint_files) > 0
            
            # Test loading checkpoint
            checkpoint_path = checkpoint_files[0]
            loaded_checkpoint = trainer.load_checkpoint(str(checkpoint_path))
            
            assert 'model_state_dict' in loaded_checkpoint
            assert 'optimizer_state_dict' in loaded_checkpoint
            
    def test_prediction(self):
        """Test model prediction functionality."""
        trainer = ExoplanetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=torch.device('cpu'),
            experiment_name="test_prediction"
        )
        
        predictions, targets = trainer.predict(self.val_loader)
        
        assert len(predictions) == len(targets)
        assert len(predictions) > 0
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
        
    def test_training_summary(self):
        """Test training summary generation."""
        trainer = ExoplanetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=torch.device('cpu'),
            experiment_name="test_summary"
        )
        
        # Train for one epoch to populate history
        trainer.train(epochs=1, verbose=False)
        
        summary = trainer.get_training_summary()
        
        assert 'experiment_name' in summary
        assert 'total_epochs' in summary
        assert 'best_epoch' in summary
        assert 'best_val_f1' in summary
        assert summary['experiment_name'] == "test_summary"


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create minimal dataset
        n_samples = 16
        sequence_length = 256
        n_channels = 2
        
        data = np.random.randn(n_samples, n_channels, sequence_length)
        labels = np.random.randint(0, 2, n_samples)
        metadata = [{'star_id': f'test_{i}'} for i in range(n_samples)]
        
        # Create dataset and loader
        dataset = LightCurveDataset(data, labels, metadata)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        # Create model and training components
        model = ExoplanetCNN(input_channels=n_channels, sequence_length=sequence_length)
        criterion = nn.BCELoss()
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        
        # Create trainer
        trainer = ExoplanetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
            experiment_name="integration_test"
        )
        
        # Train for one epoch
        history = trainer.train(epochs=1, verbose=False)
        
        # Verify training completed successfully
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1
        assert history['train_loss'][0] >= 0
        assert history['val_loss'][0] >= 0
        
    def test_gradient_accumulation(self):
        """Test that gradients are properly accumulated and applied."""
        # Create simple model and data
        model = nn.Linear(10, 1)
        x = torch.randn(4, 10, requires_grad=True)
        y = torch.randint(0, 2, (4,)).float()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.BCEWithLogitsLoss()
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass and backward pass
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break
                
        assert params_changed, "Model parameters should change after training step"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])