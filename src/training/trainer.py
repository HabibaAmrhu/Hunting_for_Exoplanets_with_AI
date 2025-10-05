"""
Training infrastructure for exoplanet detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import json
import warnings
from tqdm import tqdm

from data.types import TrainingConfig
# from utils.reproducibility import set_seed, get_device


class ExoplanetTrainer:
    """
    Handles training loop with checkpointing for exoplanet detection models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'learning_rates': []
        }
    
    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        if hasattr(optim, 'AdamW'):
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            # Fallback to Adam if AdamW not available
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def _setup_loss_function(self):
        """Setup loss function based on configuration."""
        from ..models.cnn import get_loss_function
        
        self.criterion = get_loss_function(
            loss_type=self.config.loss_function,
            class_weights=self.config.class_weights,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma
        )
        self.criterion.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, targets, metadata) in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), targets.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            # Convert outputs to predictions
            predictions = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{running_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets, metadata in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs.squeeze(), targets.float())
                
                # Track metrics
                running_loss += loss.item()
                
                # Store predictions and probabilities
                probabilities = outputs.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(float)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        epoch_metrics['loss'] = epoch_loss
        
        # Store probabilities for ROC/PR curves
        epoch_metrics['probabilities'] = np.array(all_probabilities)
        epoch_metrics['true_labels'] = np.array(all_targets)
        
        return epoch_metrics
    
    def _calculate_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """Calculate classification metrics."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Basic metrics
        tp = np.sum((predictions == 1) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        # Calculate metrics with zero division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train (uses config if None)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['train_precision'].append(train_metrics['precision'])
            self.training_history['val_precision'].append(val_metrics['precision'])
            self.training_history['train_recall'].append(train_metrics['recall'])
            self.training_history['val_recall'].append(val_metrics['recall'])
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for improvement
            improved = False
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                improved = True
                print(f"New best F1 score: {self.best_val_f1:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                if not improved:
                    improved = True
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if improved:
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {self.config.patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets, metadata in tqdm(test_loader, desc="Testing"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = outputs.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(float)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        # Calculate comprehensive metrics
        test_metrics = self._calculate_metrics(all_predictions, all_targets)
        test_metrics['probabilities'] = np.array(all_probabilities)
        test_metrics['true_labels'] = np.array(all_targets)
        
        return test_metrics
    
    def save_training_history(self, save_path: Union[str, Path]):
        """Save training history to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easy saving
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(save_path, index=False)
        
        print(f"Training history saved to {save_path}")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training results."""
        if not self.training_history['train_loss']:
            return {'status': 'No training completed'}
        
        summary = {
            'total_epochs': len(self.training_history['train_loss']),
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'final_train_f1': self.training_history['train_f1'][-1],
            'final_val_f1': self.training_history['val_f1'][-1],
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'config': self.config.__dict__
        }
        
        return summary