#!/usr/bin/env python3
"""
Comprehensive evaluation and benchmarking script for exoplanet detection models.

This script runs all advanced models on both mock and real datasets, generates
comprehensive metrics, visualizations, and performance analysis.
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Import our modules
from data.dataset import create_train_val_datasets, collate_fn
from models.cnn import ExoplanetCNN, create_loss_function
from models.lstm import ExoplanetLSTM, LightweightLSTM, create_lstm_model
from models.transformer import ExoplanetTransformer, LightweightTransformer, create_transformer_model
from models.ensemble import EnsembleModel, create_ensemble, optimize_ensemble_weights
from training.trainer import ExoplanetTrainer, create_optimizer, create_scheduler
from training.metrics import MetricsCalculator
from utils.reproducibility import set_seed
from preprocessing.synthetic_injection import SyntheticTransitInjector


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for exoplanet detection models."""
    
    def __init__(self, results_dir: Path, device: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save results
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Storage for results
        self.model_results = {}
        self.ensemble_results = {}
        self.benchmark_data = {}
    
    def create_mock_dataset(self, n_samples: int = 2000, planet_fraction: float = 0.15) -> Path:
        """Create comprehensive mock dataset for evaluation."""
        print(f"Creating mock dataset with {n_samples} samples...")
        
        data_dir = self.results_dir / 'datasets' / 'mock'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducible mock data
        np.random.seed(42)
        
        # Generate mock light curves with realistic characteristics
        light_curves = []
        labels = []
        star_ids = []
        metadata = []
        
        n_planets = int(n_samples * planet_fraction)
        n_non_planets = n_samples - n_planets
        
        sequence_length = 2048
        
        print("Generating non-planet light curves...")
        # Non-planet light curves with various stellar types
        for i in tqdm(range(n_non_planets), desc="Non-planets"):
            # Stellar parameters
            stellar_temp = np.random.uniform(3500, 7000)
            stellar_mag = np.random.uniform(10, 16)
            
            # Base stellar variability
            raw = np.random.normal(0, 0.001, sequence_length)  # Photometric noise
            
            # Add stellar variability patterns
            if np.random.random() < 0.3:  # 30% have rotational modulation
                rotation_period = np.random.uniform(5, 30)  # days
                rotation_amplitude = np.random.uniform(0.001, 0.01)
                time_array = np.arange(sequence_length) / 48  # 30-min cadence
                raw += rotation_amplitude * np.sin(2 * np.pi * time_array / rotation_period)
            
            if np.random.random() < 0.1:  # 10% have flares
                n_flares = np.random.randint(1, 4)
                for _ in range(n_flares):
                    flare_start = np.random.randint(0, sequence_length - 100)
                    flare_duration = np.random.randint(10, 50)
                    flare_amplitude = np.random.uniform(0.005, 0.02)
                    flare_profile = flare_amplitude * np.exp(-np.arange(flare_duration) / 10)
                    raw[flare_start:flare_start+len(flare_profile)] += flare_profile
            
            # Phase-folded channel (similar but with slight differences)
            phase_folded = raw + np.random.normal(0, 0.0005, sequence_length)
            
            light_curves.append(np.stack([raw, phase_folded]))
            labels.append(0)
            star_ids.append(f'star_{i:06d}')
            metadata.append({
                'star_id': f'star_{i:06d}',
                'magnitude': stellar_mag,
                'temperature': stellar_temp,
                'has_rotation': np.random.random() < 0.3,
                'has_flares': np.random.random() < 0.1
            })
        
        print("Generating planet light curves...")
        # Planet light curves with realistic transit parameters
        for i in tqdm(range(n_planets), desc="Planets"):
            # Stellar parameters
            stellar_temp = np.random.uniform(3500, 7000)
            stellar_mag = np.random.uniform(10, 16)
            stellar_radius = np.random.uniform(0.5, 2.0)  # Solar radii
            
            # Planet parameters (realistic distributions)
            planet_radius = np.random.lognormal(np.log(2.0), 0.5)  # Earth radii
            orbital_period = np.random.lognormal(np.log(10), 1.0)  # days
            impact_parameter = np.random.uniform(0, 0.9)
            
            # Calculate transit depth and duration
            transit_depth = (planet_radius / (stellar_radius * 109.2))**2  # Realistic depth
            transit_duration = np.random.uniform(2, 12)  # hours
            
            # Base stellar signal
            raw = np.random.normal(0, 0.001, sequence_length)
            
            # Add stellar variability (same as non-planets)
            if np.random.random() < 0.3:
                rotation_period = np.random.uniform(5, 30)
                rotation_amplitude = np.random.uniform(0.001, 0.01)
                time_array = np.arange(sequence_length) / 48
                raw += rotation_amplitude * np.sin(2 * np.pi * time_array / rotation_period)
            
            # Add transit signals
            duration_points = int(transit_duration * 2)  # 30-min cadence
            period_points = int(orbital_period * 48)  # Convert to data points
            
            # Calculate number of transits in observation
            n_transits = max(1, sequence_length // period_points)
            
            for j in range(n_transits):
                transit_center = j * period_points + np.random.randint(-5, 5)
                transit_start = max(0, transit_center - duration_points // 2)
                transit_end = min(sequence_length, transit_center + duration_points // 2)
                
                if transit_end > transit_start:
                    # Create realistic transit shape with limb darkening
                    transit_length = transit_end - transit_start
                    transit_profile = np.ones(transit_length)
                    
                    # Ingress/egress (20% of transit duration each)
                    ingress_points = max(1, transit_length // 5)
                    egress_points = max(1, transit_length // 5)
                    
                    # Smooth ingress
                    if ingress_points < transit_length:
                        transit_profile[:ingress_points] = np.linspace(1, 1-transit_depth, ingress_points)
                    
                    # Flat bottom
                    if ingress_points + egress_points < transit_length:
                        transit_profile[ingress_points:-egress_points] = 1 - transit_depth
                    
                    # Smooth egress
                    if egress_points < transit_length:
                        transit_profile[-egress_points:] = np.linspace(1-transit_depth, 1, egress_points)
                    
                    raw[transit_start:transit_end] *= transit_profile
            
            # Phase-folded channel with enhanced transit visibility
            phase_folded = raw.copy()
            # Slight enhancement for phase-folding effect
            for j in range(n_transits):
                transit_center = j * period_points
                transit_start = max(0, transit_center - duration_points // 2)
                transit_end = min(sequence_length, transit_center + duration_points // 2)
                if transit_end > transit_start:
                    phase_folded[transit_start:transit_end] *= 0.995  # Slight additional dip
            
            light_curves.append(np.stack([raw, phase_folded]))
            labels.append(1)
            star_ids.append(f'planet_{i:06d}')
            metadata.append({
                'star_id': f'planet_{i:06d}',
                'magnitude': stellar_mag,
                'temperature': stellar_temp,
                'planet_radius': planet_radius,
                'orbital_period': orbital_period,
                'transit_depth': transit_depth,
                'transit_duration': transit_duration,
                'impact_parameter': impact_parameter
            })
        
        # Save dataset
        dataset_path = data_dir / 'comprehensive_dataset.npz'
        np.savez_compressed(
            dataset_path,
            light_curves=np.array(light_curves),
            labels=np.array(labels),
            star_ids=np.array(star_ids),
            metadata=np.array(metadata)
        )
        
        # Save metadata CSV
        df = pd.DataFrame(metadata)
        df['label'] = labels
        df.to_csv(data_dir / 'metadata.csv', index=False)
        
        print(f"Dataset created: {len(light_curves)} samples")
        print(f"Class distribution: {np.bincount(labels)}")
        print(f"Planet fraction: {np.mean(labels):.3f}")
        
        return dataset_path    

    def train_and_evaluate_model(
        self, 
        model: nn.Module, 
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train and comprehensively evaluate a single model."""
        print(f"\n{'='*60}")
        print(f"Training and Evaluating: {model_name}")
        print(f"Parameters: {model.count_parameters():,}")
        print(f"{'='*60}")
        
        model_dir = self.results_dir / 'individual_models' / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training setup
        criterion = create_loss_function('focal', alpha=0.25, gamma=2.0)
        optimizer = create_optimizer(model, 'adamw', config['learning_rate'], weight_decay=0.01)
        scheduler = create_scheduler(optimizer, 'cosine', T_max=config['epochs'])
        
        # Create trainer
        trainer = ExoplanetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            checkpoint_dir=str(model_dir),
            experiment_name=model_name
        )
        
        # Training
        start_time = time.time()
        history = trainer.train(
            epochs=config['epochs'],
            patience=config['patience'],
            save_best=True,
            verbose=True
        )
        training_time = time.time() - start_time
        
        # Load best model for evaluation
        best_checkpoint = model_dir / f"{model_name}_best.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # Comprehensive evaluation
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'config': config,
            'history': history
        }
        
        # Evaluate on validation and test sets
        for split_name, data_loader in [('validation', val_loader), ('test', test_loader)]:
            print(f"Evaluating on {split_name} set...")
            
            # Get predictions
            predictions, targets = trainer.predict(data_loader)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_metrics(targets, predictions)
            optimal_threshold, optimal_f1 = self.metrics_calc.find_optimal_threshold(
                targets, predictions, 'f1'
            )
            optimal_metrics = self.metrics_calc.calculate_metrics(
                targets, predictions, optimal_threshold
            )
            
            # Create comprehensive report
            report = self.metrics_calc.create_comprehensive_report(
                targets, predictions,
                model_name=f"{model_name}_{split_name}",
                save_dir=str(model_dir / split_name)
            )
            
            results[f'{split_name}_metrics'] = metrics
            results[f'{split_name}_optimal_metrics'] = optimal_metrics
            results[f'{split_name}_optimal_threshold'] = optimal_threshold
            results[f'{split_name}_predictions'] = predictions
            results[f'{split_name}_targets'] = targets
            results[f'{split_name}_report'] = report
        
        # Attention analysis (if supported)
        if hasattr(model, 'get_attention_weights'):
            print("Analyzing attention patterns...")
            attention_analysis = self.analyze_attention_patterns(
                model, test_loader, model_dir / 'attention_analysis'
            )
            results['attention_analysis'] = attention_analysis
        
        # Efficiency analysis
        efficiency_metrics = self.analyze_model_efficiency(model, test_loader)
        results['efficiency_metrics'] = efficiency_metrics
        
        # Save results
        with open(model_dir / 'comprehensive_results.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        # Save predictions for ensemble
        with open(model_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump({
                'val_predictions': results['validation_predictions'],
                'val_targets': results['validation_targets'],
                'test_predictions': results['test_predictions'],
                'test_targets': results['test_targets']
            }, f)
        
        print(f"Best validation F1: {results['validation_optimal_metrics']['f1_score']:.4f}")
        print(f"Test F1: {results['test_optimal_metrics']['f1_score']:.4f}")
        print(f"Training time: {training_time:.1f}s")
        
        return results
    
    def analyze_attention_patterns(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        save_dir: Path
    ) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        attention_data = []
        
        # Sample a few examples for attention analysis
        sample_count = 0
        max_samples = 20
        
        with torch.no_grad():
            for data, targets, metadata in data_loader:
                if sample_count >= max_samples:
                    break
                
                data = data.to(self.device)
                
                # Get attention weights
                try:
                    attention_weights = model.get_attention_weights(data)
                    if attention_weights is not None:
                        for i in range(min(data.size(0), max_samples - sample_count)):
                            attention_data.append({
                                'sample_idx': sample_count + i,
                                'label': targets[i].item(),
                                'attention_weights': attention_weights[i].cpu().numpy(),
                                'metadata': metadata[i] if metadata else {}
                            })
                        sample_count += data.size(0)
                except Exception as e:
                    print(f"Could not extract attention weights: {e}")
                    break
        
        if not attention_data:
            return {'error': 'No attention weights could be extracted'}
        
        # Create attention visualizations
        self.create_attention_visualizations(attention_data, save_dir)
        
        # Analyze attention patterns
        analysis = {
            'n_samples': len(attention_data),
            'avg_attention_entropy': np.mean([
                -np.sum(att['attention_weights'] * np.log(att['attention_weights'] + 1e-8))
                for att in attention_data
            ]),
            'attention_focus_regions': self.identify_attention_focus_regions(attention_data)
        }
        
        return analysis
    
    def create_attention_visualizations(self, attention_data: List[Dict], save_dir: Path):
        """Create attention visualization plots."""
        # Plot attention patterns for positive and negative examples
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Separate by label
        positive_examples = [att for att in attention_data if att['label'] == 1]
        negative_examples = [att for att in attention_data if att['label'] == 0]
        
        # Plot average attention for each class
        if positive_examples:
            avg_pos_attention = np.mean([att['attention_weights'] for att in positive_examples], axis=0)
            axes[0, 0].plot(avg_pos_attention)
            axes[0, 0].set_title('Average Attention - Positive Examples (Planets)')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Attention Weight')
        
        if negative_examples:
            avg_neg_attention = np.mean([att['attention_weights'] for att in negative_examples], axis=0)
            axes[0, 1].plot(avg_neg_attention)
            axes[0, 1].set_title('Average Attention - Negative Examples (No Planets)')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Attention Weight')
        
        # Plot individual examples
        if positive_examples:
            for i, att in enumerate(positive_examples[:3]):
                axes[1, 0].plot(att['attention_weights'], alpha=0.7, label=f'Example {i+1}')
            axes[1, 0].set_title('Individual Positive Examples')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Attention Weight')
            axes[1, 0].legend()
        
        if negative_examples:
            for i, att in enumerate(negative_examples[:3]):
                axes[1, 1].plot(att['attention_weights'], alpha=0.7, label=f'Example {i+1}')
            axes[1, 1].set_title('Individual Negative Examples')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Attention Weight')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'attention_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create attention heatmap
        if len(attention_data) > 1:
            attention_matrix = np.array([att['attention_weights'] for att in attention_data])
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention_matrix, cmap='viridis', cbar=True)
            plt.title('Attention Patterns Across Samples')
            plt.xlabel('Time Step')
            plt.ylabel('Sample')
            plt.savefig(save_dir / 'attention_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def identify_attention_focus_regions(self, attention_data: List[Dict]) -> Dict[str, Any]:
        """Identify regions where attention typically focuses."""
        all_attention = np.array([att['attention_weights'] for att in attention_data])
        
        # Find peak attention regions
        avg_attention = np.mean(all_attention, axis=0)
        attention_threshold = np.percentile(avg_attention, 90)  # Top 10% attention
        
        focus_regions = []
        in_region = False
        region_start = 0
        
        for i, att_val in enumerate(avg_attention):
            if att_val > attention_threshold and not in_region:
                region_start = i
                in_region = True
            elif att_val <= attention_threshold and in_region:
                focus_regions.append((region_start, i))
                in_region = False
        
        if in_region:  # Handle case where region extends to end
            focus_regions.append((region_start, len(avg_attention)))
        
        return {
            'focus_regions': focus_regions,
            'avg_attention_peak': float(np.max(avg_attention)),
            'attention_concentration': float(np.std(avg_attention))
        }
    
    def analyze_model_efficiency(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        """Analyze model computational efficiency."""
        model.eval()
        
        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Inference timing
        inference_times = []
        batch_sizes = []
        
        with torch.no_grad():
            for i, (data, _, _) in enumerate(data_loader):
                if i >= 10:  # Test on first 10 batches
                    break
                
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Time inference
                start_time = time.time()
                _ = model(data)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                batch_sizes.append(batch_size)
        
        # Calculate efficiency metrics
        avg_inference_time = np.mean(inference_times)
        avg_batch_size = np.mean(batch_sizes)
        samples_per_second = avg_batch_size / avg_inference_time
        
        efficiency_metrics = {
            'avg_inference_time_per_batch': float(avg_inference_time),
            'avg_batch_size': float(avg_batch_size),
            'samples_per_second': float(samples_per_second),
            'ms_per_sample': float(avg_inference_time * 1000 / avg_batch_size),
            'model_parameters': model.count_parameters() if hasattr(model, 'count_parameters') else 0
        }
        
        # GPU memory usage (if available)
        if torch.cuda.is_available():
            efficiency_metrics['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return efficiency_metrics   
 
    def create_and_evaluate_ensembles(
        self,
        individual_results: List[Dict[str, Any]],
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Create and evaluate ensemble models."""
        print(f"\n{'='*60}")
        print("Creating and Evaluating Ensemble Models")
        print(f"{'='*60}")
        
        ensemble_dir = self.results_dir / 'ensembles'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Load individual model predictions
        model_predictions = {}
        model_targets = {}
        
        for result in individual_results:
            model_name = result['model_name']
            model_dir = self.results_dir / 'individual_models' / model_name
            
            try:
                with open(model_dir / 'predictions.pkl', 'rb') as f:
                    pred_data = pickle.load(f)
                    model_predictions[model_name] = pred_data
                    model_targets[model_name] = pred_data['val_targets']  # Should be same for all
                print(f"Loaded predictions for {model_name}")
            except Exception as e:
                print(f"Could not load predictions for {model_name}: {e}")
        
        if len(model_predictions) < 2:
            print("Need at least 2 models for ensemble. Skipping ensemble evaluation.")
            return {}
        
        # Get common targets (should be identical across models)
        val_targets = list(model_targets.values())[0]
        test_targets = model_predictions[list(model_predictions.keys())[0]]['test_targets']
        
        ensemble_results = {}
        
        # Test different ensemble methods
        ensemble_methods = [
            'weighted_average',
            'voting',
            'learned'  # If we have enough data
        ]
        
        for method in ensemble_methods:
            print(f"\nEvaluating ensemble method: {method}")
            
            method_dir = ensemble_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ensemble predictions for validation set
            val_predictions_matrix = np.array([
                model_predictions[name]['val_predictions'] 
                for name in model_predictions.keys()
            ]).T  # Shape: (n_samples, n_models)
            
            # Create ensemble predictions for test set
            test_predictions_matrix = np.array([
                model_predictions[name]['test_predictions'] 
                for name in model_predictions.keys()
            ]).T  # Shape: (n_samples, n_models)
            
            if method == 'weighted_average':
                # Optimize weights on validation set
                best_weights = self.optimize_ensemble_weights_grid_search(
                    val_predictions_matrix, val_targets
                )
                
                val_ensemble_pred = np.average(val_predictions_matrix, axis=1, weights=best_weights)
                test_ensemble_pred = np.average(test_predictions_matrix, axis=1, weights=best_weights)
                
                ensemble_info = {
                    'method': method,
                    'weights': best_weights.tolist(),
                    'model_names': list(model_predictions.keys())
                }
                
            elif method == 'voting':
                # Simple majority voting
                val_binary_preds = (val_predictions_matrix > 0.5).astype(int)
                test_binary_preds = (test_predictions_matrix > 0.5).astype(int)
                
                val_votes = np.sum(val_binary_preds, axis=1)
                test_votes = np.sum(test_binary_preds, axis=1)
                
                n_models = val_predictions_matrix.shape[1]
                val_ensemble_pred = (val_votes > n_models / 2).astype(float)
                test_ensemble_pred = (test_votes > n_models / 2).astype(float)
                
                ensemble_info = {
                    'method': method,
                    'n_models': n_models,
                    'model_names': list(model_predictions.keys())
                }
                
            elif method == 'learned':
                # Simple learned combination (linear combination with learned weights)
                from sklearn.linear_model import LogisticRegression
                
                # Train meta-learner on validation predictions
                meta_learner = LogisticRegression(random_state=42)
                meta_learner.fit(val_predictions_matrix, val_targets)
                
                val_ensemble_pred = meta_learner.predict_proba(val_predictions_matrix)[:, 1]
                test_ensemble_pred = meta_learner.predict_proba(test_predictions_matrix)[:, 1]
                
                ensemble_info = {
                    'method': method,
                    'meta_learner_coef': meta_learner.coef_[0].tolist(),
                    'meta_learner_intercept': float(meta_learner.intercept_[0]),
                    'model_names': list(model_predictions.keys())
                }
            
            # Calculate ensemble uncertainty (standard deviation across models)
            val_uncertainty = np.std(val_predictions_matrix, axis=1)
            test_uncertainty = np.std(test_predictions_matrix, axis=1)
            
            # Evaluate ensemble performance
            val_metrics = self.metrics_calc.calculate_metrics(val_targets, val_ensemble_pred)
            val_optimal_threshold, val_optimal_f1 = self.metrics_calc.find_optimal_threshold(
                val_targets, val_ensemble_pred, 'f1'
            )
            val_optimal_metrics = self.metrics_calc.calculate_metrics(
                val_targets, val_ensemble_pred, val_optimal_threshold
            )
            
            test_metrics = self.metrics_calc.calculate_metrics(test_targets, test_ensemble_pred)
            test_optimal_metrics = self.metrics_calc.calculate_metrics(
                test_targets, test_ensemble_pred, val_optimal_threshold  # Use val threshold
            )
            
            # Create comprehensive reports
            val_report = self.metrics_calc.create_comprehensive_report(
                val_targets, val_ensemble_pred,
                model_name=f"Ensemble_{method}_validation",
                save_dir=str(method_dir / 'validation')
            )
            
            test_report = self.metrics_calc.create_comprehensive_report(
                test_targets, test_ensemble_pred,
                model_name=f"Ensemble_{method}_test",
                save_dir=str(method_dir / 'test')
            )
            
            # Store results
            ensemble_result = {
                'method': method,
                'ensemble_info': ensemble_info,
                'validation_metrics': val_metrics,
                'validation_optimal_metrics': val_optimal_metrics,
                'validation_optimal_threshold': val_optimal_threshold,
                'test_metrics': test_metrics,
                'test_optimal_metrics': test_optimal_metrics,
                'validation_uncertainty': {
                    'mean': float(np.mean(val_uncertainty)),
                    'std': float(np.std(val_uncertainty)),
                    'median': float(np.median(val_uncertainty))
                },
                'test_uncertainty': {
                    'mean': float(np.mean(test_uncertainty)),
                    'std': float(np.std(test_uncertainty)),
                    'median': float(np.median(test_uncertainty))
                },
                'validation_report': val_report,
                'test_report': test_report
            }
            
            ensemble_results[method] = ensemble_result
            
            # Save ensemble results
            with open(method_dir / 'ensemble_results.json', 'w') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    return obj
                
                json.dump(ensemble_result, f, indent=2, default=convert_numpy)
            
            print(f"Ensemble {method} - Val F1: {val_optimal_f1:.4f}, Test F1: {test_optimal_metrics['f1_score']:.4f}")
        
        return ensemble_results
    
    def optimize_ensemble_weights_grid_search(
        self, 
        predictions_matrix: np.ndarray, 
        targets: np.ndarray
    ) -> np.ndarray:
        """Optimize ensemble weights using grid search."""
        from sklearn.metrics import f1_score
        
        n_models = predictions_matrix.shape[1]
        
        if n_models == 2:
            # For 2 models, search over weight for first model
            best_f1 = 0
            best_weights = np.array([0.5, 0.5])
            
            for w1 in np.linspace(0.1, 0.9, 9):
                w2 = 1 - w1
                weights = np.array([w1, w2])
                
                ensemble_pred = np.average(predictions_matrix, axis=1, weights=weights)
                binary_pred = (ensemble_pred > 0.5).astype(int)
                f1 = f1_score(targets, binary_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = weights
        
        elif n_models == 3:
            # For 3 models, search over simplex
            best_f1 = 0
            best_weights = np.array([1/3, 1/3, 1/3])
            
            for w1 in np.linspace(0.1, 0.8, 8):
                for w2 in np.linspace(0.1, 0.9 - w1, 8):
                    w3 = 1 - w1 - w2
                    if w3 >= 0.1:
                        weights = np.array([w1, w2, w3])
                        
                        ensemble_pred = np.average(predictions_matrix, axis=1, weights=weights)
                        binary_pred = (ensemble_pred > 0.5).astype(int)
                        f1 = f1_score(targets, binary_pred)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = weights
        
        else:
            # For more models, use random search
            best_f1 = 0
            best_weights = np.ones(n_models) / n_models
            
            for _ in range(100):
                # Generate random weights
                weights = np.random.dirichlet(np.ones(n_models))
                
                ensemble_pred = np.average(predictions_matrix, axis=1, weights=weights)
                binary_pred = (ensemble_pred > 0.5).astype(int)
                f1 = f1_score(targets, binary_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = weights
        
        return best_weights 
   
    def create_comprehensive_comparison(
        self,
        individual_results: List[Dict[str, Any]],
        ensemble_results: Dict[str, Any]
    ):
        """Create comprehensive comparison of all models."""
        print(f"\n{'='*60}")
        print("Creating Comprehensive Model Comparison")
        print(f"{'='*60}")
        
        comparison_dir = self.results_dir / 'comprehensive_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare comparison data
        comparison_data = []
        
        # Individual models
        for result in individual_results:
            model_info = result.get('model_info', {})
            efficiency = result.get('efficiency_metrics', {})
            
            comparison_data.append({
                'Model': result['model_name'],
                'Type': 'Individual',
                'Architecture': model_info.get('model_name', result['model_name']),
                'Parameters': model_info.get('total_parameters', 0),
                'Val_F1': result['validation_optimal_metrics']['f1_score'],
                'Test_F1': result['test_optimal_metrics']['f1_score'],
                'Val_Precision': result['validation_optimal_metrics']['precision'],
                'Test_Precision': result['test_optimal_metrics']['precision'],
                'Val_Recall': result['validation_optimal_metrics']['recall'],
                'Test_Recall': result['test_optimal_metrics']['recall'],
                'Val_ROC_AUC': result['validation_optimal_metrics']['roc_auc'],
                'Test_ROC_AUC': result['test_optimal_metrics']['roc_auc'],
                'Training_Time': result['training_time'],
                'Inference_Speed': efficiency.get('samples_per_second', 0),
                'Memory_MB': efficiency.get('peak_gpu_memory_mb', 0)
            })
        
        # Ensemble models
        for method, result in ensemble_results.items():
            comparison_data.append({
                'Model': f'Ensemble_{method}',
                'Type': 'Ensemble',
                'Architecture': f'Ensemble ({method})',
                'Parameters': sum([r['model_info'].get('total_parameters', 0) for r in individual_results]),
                'Val_F1': result['validation_optimal_metrics']['f1_score'],
                'Test_F1': result['test_optimal_metrics']['f1_score'],
                'Val_Precision': result['validation_optimal_metrics']['precision'],
                'Test_Precision': result['test_optimal_metrics']['precision'],
                'Val_Recall': result['validation_optimal_metrics']['recall'],
                'Test_Recall': result['test_optimal_metrics']['recall'],
                'Val_ROC_AUC': result['validation_optimal_metrics']['roc_auc'],
                'Test_ROC_AUC': result['test_optimal_metrics']['roc_auc'],
                'Training_Time': 0,  # Ensemble doesn't have training time
                'Inference_Speed': 0,  # Would need to measure separately
                'Memory_MB': 0
            })
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test_F1', ascending=False)
        
        # Save comparison table
        df.to_csv(comparison_dir / 'model_comparison.csv', index=False)
        
        # Print comparison table
        print("\nModel Performance Comparison:")
        print("="*120)
        display_cols = ['Model', 'Type', 'Parameters', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Test_ROC_AUC', 'Training_Time']
        print(df[display_cols].to_string(index=False, float_format='%.4f'))
        
        # Create comprehensive visualizations
        self.create_performance_visualizations(df, comparison_dir)
        self.create_efficiency_analysis(df, comparison_dir)
        self.create_uncertainty_analysis(individual_results, ensemble_results, comparison_dir)
        
        # Generate insights
        insights = self.generate_performance_insights(df, individual_results, ensemble_results)
        
        # Save insights
        with open(comparison_dir / 'performance_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        return df, insights
    
    def create_performance_visualizations(self, df: pd.DataFrame, save_dir: Path):
        """Create comprehensive performance visualization plots."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. F1 Score Comparison
        ax1 = plt.subplot(3, 4, 1)
        models = df['Model'].tolist()
        test_f1 = df['Test_F1'].tolist()
        colors = ['lightblue' if t == 'Individual' else 'lightgreen' for t in df['Type']]
        
        bars = ax1.bar(range(len(models)), test_f1, color=colors)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Test F1 Score')
        ax1.set_title('Test F1 Score Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, f1 in zip(bars, test_f1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. ROC AUC Comparison
        ax2 = plt.subplot(3, 4, 2)
        test_roc = df['Test_ROC_AUC'].tolist()
        ax2.bar(range(len(models)), test_roc, color=colors)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Test ROC AUC')
        ax2.set_title('ROC AUC Comparison')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # 3. Precision vs Recall
        ax3 = plt.subplot(3, 4, 3)
        test_precision = df['Test_Precision'].tolist()
        test_recall = df['Test_Recall'].tolist()
        
        for i, (prec, rec, model, model_type) in enumerate(zip(test_precision, test_recall, models, df['Type'])):
            color = 'blue' if model_type == 'Individual' else 'green'
            marker = 'o' if model_type == 'Individual' else 's'
            ax3.scatter(rec, prec, s=100, color=color, marker=marker, alpha=0.7, label=model if i < 6 else "")
        
        ax3.set_xlabel('Test Recall')
        ax3.set_ylabel('Test Precision')
        ax3.set_title('Precision vs Recall')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Complexity vs Performance
        ax4 = plt.subplot(3, 4, 4)
        individual_df = df[df['Type'] == 'Individual']
        if len(individual_df) > 0:
            params = individual_df['Parameters'].tolist()
            f1_scores = individual_df['Test_F1'].tolist()
            model_names = individual_df['Model'].tolist()
            
            ax4.scatter(params, f1_scores, s=100, alpha=0.7)
            for i, name in enumerate(model_names):
                ax4.annotate(name, (params[i], f1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax4.set_xlabel('Parameters')
            ax4.set_ylabel('Test F1 Score')
            ax4.set_title('Model Complexity vs Performance')
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3)
        
        # 5. Training Time vs Performance
        ax5 = plt.subplot(3, 4, 5)
        individual_df = df[df['Type'] == 'Individual']
        if len(individual_df) > 0:
            training_times = individual_df['Training_Time'].tolist()
            f1_scores = individual_df['Test_F1'].tolist()
            model_names = individual_df['Model'].tolist()
            
            ax5.scatter(training_times, f1_scores, s=100, alpha=0.7)
            for i, name in enumerate(model_names):
                ax5.annotate(name, (training_times[i], f1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax5.set_xlabel('Training Time (s)')
            ax5.set_ylabel('Test F1 Score')
            ax5.set_title('Training Time vs Performance')
            ax5.grid(True, alpha=0.3)
        
        # 6. Validation vs Test Performance
        ax6 = plt.subplot(3, 4, 6)
        val_f1 = df['Val_F1'].tolist()
        test_f1 = df['Test_F1'].tolist()
        
        ax6.scatter(val_f1, test_f1, s=100, alpha=0.7)
        ax6.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect correlation line
        
        for i, model in enumerate(models):
            ax6.annotate(model, (val_f1[i], test_f1[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('Validation F1')
        ax6.set_ylabel('Test F1')
        ax6.set_title('Validation vs Test Performance')
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance Improvement Analysis
        ax7 = plt.subplot(3, 4, 7)
        if len(df) > 1:
            baseline_f1 = df.iloc[-1]['Test_F1']  # Assume worst performing is baseline
            improvements = [(f1 - baseline_f1) / baseline_f1 * 100 for f1 in test_f1]
            
            bars = ax7.bar(range(len(models)), improvements, color=colors)
            ax7.set_xlabel('Model')
            ax7.set_ylabel('F1 Improvement (%)')
            ax7.set_title('Performance Improvement over Baseline')
            ax7.set_xticks(range(len(models)))
            ax7.set_xticklabels(models, rotation=45, ha='right')
            ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 8. Efficiency Analysis
        ax8 = plt.subplot(3, 4, 8)
        individual_df = df[df['Type'] == 'Individual']
        if len(individual_df) > 0 and 'Inference_Speed' in individual_df.columns:
            inference_speeds = individual_df['Inference_Speed'].tolist()
            f1_scores = individual_df['Test_F1'].tolist()
            
            if any(speed > 0 for speed in inference_speeds):
                ax8.scatter(inference_speeds, f1_scores, s=100, alpha=0.7)
                ax8.set_xlabel('Inference Speed (samples/s)')
                ax8.set_ylabel('Test F1 Score')
                ax8.set_title('Inference Speed vs Performance')
                ax8.grid(True, alpha=0.3)
        
        # 9-12. Individual model performance breakdown
        metrics = ['Test_F1', 'Test_Precision', 'Test_Recall', 'Test_ROC_AUC']
        for i, metric in enumerate(metrics):
            ax = plt.subplot(3, 4, 9 + i)
            values = df[metric].tolist()
            
            ax.barh(range(len(models)), values, color=colors)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            ax.set_xlabel(metric.replace('Test_', '').replace('_', ' '))
            ax.set_title(f'{metric.replace("Test_", "").replace("_", " ")} Comparison')
            
            # Add value labels
            for j, val in enumerate(values):
                ax.text(val + 0.01, j, f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comprehensive_performance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_efficiency_analysis(self, df: pd.DataFrame, save_dir: Path):
        """Create efficiency analysis plots."""
        individual_df = df[df['Type'] == 'Individual'].copy()
        
        if len(individual_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Parameters vs Performance
        if 'Parameters' in individual_df.columns:
            axes[0, 0].scatter(individual_df['Parameters'], individual_df['Test_F1'], s=100, alpha=0.7)
            axes[0, 0].set_xlabel('Model Parameters')
            axes[0, 0].set_ylabel('Test F1 Score')
            axes[0, 0].set_title('Model Size vs Performance')
            axes[0, 0].set_xscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            for i, row in individual_df.iterrows():
                axes[0, 0].annotate(row['Model'], (row['Parameters'], row['Test_F1']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Training Time vs Performance
        if 'Training_Time' in individual_df.columns:
            axes[0, 1].scatter(individual_df['Training_Time'], individual_df['Test_F1'], s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Training Time (s)')
            axes[0, 1].set_ylabel('Test F1 Score')
            axes[0, 1].set_title('Training Efficiency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Efficiency Score (F1 per parameter)
        if 'Parameters' in individual_df.columns:
            efficiency_scores = individual_df['Test_F1'] / (individual_df['Parameters'] / 1e6)  # F1 per million parameters
            axes[1, 0].bar(range(len(individual_df)), efficiency_scores)
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('F1 per Million Parameters')
            axes[1, 0].set_title('Parameter Efficiency')
            axes[1, 0].set_xticks(range(len(individual_df)))
            axes[1, 0].set_xticklabels(individual_df['Model'], rotation=45, ha='right')
        
        # Training Efficiency (F1 per training minute)
        if 'Training_Time' in individual_df.columns:
            training_efficiency = individual_df['Test_F1'] / (individual_df['Training_Time'] / 60)  # F1 per minute
            axes[1, 1].bar(range(len(individual_df)), training_efficiency)
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('F1 per Training Minute')
            axes[1, 1].set_title('Training Efficiency')
            axes[1, 1].set_xticks(range(len(individual_df)))
            axes[1, 1].set_xticklabels(individual_df['Model'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_uncertainty_analysis(
        self,
        individual_results: List[Dict[str, Any]],
        ensemble_results: Dict[str, Any],
        save_dir: Path
    ):
        """Create uncertainty analysis plots."""
        if not ensemble_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Ensemble uncertainty distributions
        ax = axes[0, 0]
        for method, result in ensemble_results.items():
            test_uncertainty = result.get('test_uncertainty', {})
            if 'mean' in test_uncertainty:
                # Create a simple bar plot for uncertainty metrics
                uncertainty_metrics = ['mean', 'std', 'median']
                values = [test_uncertainty.get(metric, 0) for metric in uncertainty_metrics]
                
                x_pos = np.arange(len(uncertainty_metrics))
                ax.bar(x_pos + len(ensemble_results.keys()) * 0.1, values, 
                      width=0.1, label=method, alpha=0.7)
        
        ax.set_xlabel('Uncertainty Metric')
        ax.set_ylabel('Value')
        ax.set_title('Ensemble Uncertainty Analysis')
        ax.set_xticks(np.arange(len(uncertainty_metrics)))
        ax.set_xticklabels(uncertainty_metrics)
        ax.legend()
        
        # Performance vs Uncertainty trade-off
        ax = axes[0, 1]
        for method, result in ensemble_results.items():
            test_f1 = result['test_optimal_metrics']['f1_score']
            test_uncertainty_mean = result.get('test_uncertainty', {}).get('mean', 0)
            
            ax.scatter(test_uncertainty_mean, test_f1, s=100, label=method, alpha=0.7)
        
        ax.set_xlabel('Mean Prediction Uncertainty')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Performance vs Uncertainty Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ensemble method comparison
        ax = axes[1, 0]
        methods = list(ensemble_results.keys())
        f1_scores = [ensemble_results[method]['test_optimal_metrics']['f1_score'] for method in methods]
        
        bars = ax.bar(methods, f1_scores, alpha=0.7)
        ax.set_xlabel('Ensemble Method')
        ax.set_ylabel('Test F1 Score')
        ax.set_title('Ensemble Method Comparison')
        
        # Add value labels
        for bar, f1 in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{f1:.3f}', ha='center', va='bottom')
        
        # Individual vs Ensemble comparison
        ax = axes[1, 1]
        
        # Get best individual model performance
        best_individual_f1 = max([r['test_optimal_metrics']['f1_score'] for r in individual_results])
        ensemble_f1s = [ensemble_results[method]['test_optimal_metrics']['f1_score'] for method in methods]
        
        # Calculate improvements
        improvements = [(f1 - best_individual_f1) / best_individual_f1 * 100 for f1 in ensemble_f1s]
        
        bars = ax.bar(methods, improvements, alpha=0.7)
        ax.set_xlabel('Ensemble Method')
        ax.set_ylabel('F1 Improvement over Best Individual (%)')
        ax.set_title('Ensemble Improvement Analysis')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{imp:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_insights(
        self,
        df: pd.DataFrame,
        individual_results: List[Dict[str, Any]],
        ensemble_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive performance insights."""
        insights = {
            'summary': {},
            'individual_models': {},
            'ensemble_analysis': {},
            'efficiency_analysis': {},
            'recommendations': []
        }
        
        # Summary statistics
        insights['summary'] = {
            'total_models_evaluated': len(df),
            'individual_models': len(df[df['Type'] == 'Individual']),
            'ensemble_models': len(df[df['Type'] == 'Ensemble']),
            'best_overall_f1': float(df['Test_F1'].max()),
            'best_model': df.loc[df['Test_F1'].idxmax(), 'Model'],
            'f1_range': {
                'min': float(df['Test_F1'].min()),
                'max': float(df['Test_F1'].max()),
                'std': float(df['Test_F1'].std())
            }
        }
        
        # Individual model analysis
        individual_df = df[df['Type'] == 'Individual']
        if len(individual_df) > 0:
            insights['individual_models'] = {
                'best_individual': individual_df.loc[individual_df['Test_F1'].idxmax(), 'Model'],
                'best_individual_f1': float(individual_df['Test_F1'].max()),
                'parameter_range': {
                    'min': int(individual_df['Parameters'].min()) if 'Parameters' in individual_df.columns else 0,
                    'max': int(individual_df['Parameters'].max()) if 'Parameters' in individual_df.columns else 0
                },
                'training_time_range': {
                    'min': float(individual_df['Training_Time'].min()) if 'Training_Time' in individual_df.columns else 0,
                    'max': float(individual_df['Training_Time'].max()) if 'Training_Time' in individual_df.columns else 0
                }
            }
        
        # Ensemble analysis
        if ensemble_results:
            best_ensemble_method = max(ensemble_results.keys(), 
                                     key=lambda k: ensemble_results[k]['test_optimal_metrics']['f1_score'])
            best_ensemble_f1 = ensemble_results[best_ensemble_method]['test_optimal_metrics']['f1_score']
            
            best_individual_f1 = individual_df['Test_F1'].max() if len(individual_df) > 0 else 0
            ensemble_improvement = (best_ensemble_f1 - best_individual_f1) / best_individual_f1 * 100
            
            insights['ensemble_analysis'] = {
                'best_ensemble_method': best_ensemble_method,
                'best_ensemble_f1': float(best_ensemble_f1),
                'improvement_over_individual': float(ensemble_improvement),
                'ensemble_methods_tested': list(ensemble_results.keys()),
                'uncertainty_quantification': {
                    method: result.get('test_uncertainty', {})
                    for method, result in ensemble_results.items()
                }
            }
        
        # Efficiency analysis
        if len(individual_df) > 0 and 'Parameters' in individual_df.columns:
            # Calculate efficiency metrics
            param_efficiency = individual_df['Test_F1'] / (individual_df['Parameters'] / 1e6)
            time_efficiency = individual_df['Test_F1'] / (individual_df['Training_Time'] / 60) if 'Training_Time' in individual_df.columns else None
            
            most_param_efficient = individual_df.loc[param_efficiency.idxmax(), 'Model']
            most_time_efficient = individual_df.loc[time_efficiency.idxmax(), 'Model'] if time_efficiency is not None else None
            
            insights['efficiency_analysis'] = {
                'most_parameter_efficient': most_param_efficient,
                'most_time_efficient': most_time_efficient,
                'parameter_efficiency_scores': param_efficiency.to_dict(),
                'time_efficiency_scores': time_efficiency.to_dict() if time_efficiency is not None else {}
            }
        
        # Generate recommendations
        recommendations = []
        
        if len(individual_df) > 0:
            best_individual = insights['individual_models']['best_individual']
            recommendations.append(f"Best individual model: {best_individual}")
        
        if ensemble_results:
            best_ensemble = insights['ensemble_analysis']['best_ensemble_method']
            improvement = insights['ensemble_analysis']['improvement_over_individual']
            recommendations.append(f"Best ensemble method: {best_ensemble} (+{improvement:.1f}% improvement)")
        
        if 'efficiency_analysis' in insights:
            efficient_model = insights['efficiency_analysis']['most_parameter_efficient']
            recommendations.append(f"Most parameter-efficient model: {efficient_model}")
        
        # Performance-based recommendations
        if insights['summary']['best_overall_f1'] > 0.9:
            recommendations.append("Excellent performance achieved (F1 > 0.9)")
        elif insights['summary']['best_overall_f1'] > 0.85:
            recommendations.append("Good performance achieved (F1 > 0.85)")
        else:
            recommendations.append("Consider additional model tuning or data augmentation")
        
        insights['recommendations'] = recommendations
        
        return insights 
   
    def run_comprehensive_evaluation(
        self,
        dataset_path: Path = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run the complete comprehensive evaluation pipeline."""
        if config is None:
            config = {
                'epochs': 15,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 5,
                'val_split': 0.2,
                'test_split': 0.1,
                'synthetic_ratio': 0.3
            }
        
        print(f"{'='*80}")
        print("COMPREHENSIVE EXOPLANET DETECTION MODEL EVALUATION")
        print(f"{'='*80}")
        
        # Create dataset if not provided
        if dataset_path is None:
            dataset_path = self.create_mock_dataset(n_samples=2000, planet_fraction=0.15)
        
        # Create datasets with train/val/test split
        print("\nCreating datasets...")
        
        # Use synthetic injection for training
        injector = SyntheticTransitInjector()
        
        # Create train/val split first
        train_dataset, temp_dataset = create_train_val_datasets(
            dataset_path,
            val_split=config['val_split'] + config['test_split'],
            use_augmentation=True,
            synthetic_injector=injector,
            synthetic_ratio=config['synthetic_ratio']
        )
        
        # Split temp_dataset into val and test
        temp_size = len(temp_dataset)
        val_size = int(temp_size * config['val_split'] / (config['val_split'] + config['test_split']))
        test_size = temp_size - val_size
        
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], 
            shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], 
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'], 
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Define models to evaluate
        models_to_evaluate = [
            {
                'name': 'cnn_baseline',
                'model': ExoplanetCNN(
                    input_channels=2, sequence_length=2048,
                    dropout_rate=0.5, use_batch_norm=True
                )
            },
            {
                'name': 'lstm_full',
                'model': create_lstm_model(
                    model_type='full', input_channels=2, sequence_length=2048,
                    config={'lstm_hidden_size': 128, 'lstm_num_layers': 2, 
                           'use_attention': True, 'dropout_rate': 0.3}
                )
            },
            {
                'name': 'lstm_lightweight',
                'model': create_lstm_model(
                    model_type='lightweight', input_channels=2, sequence_length=2048,
                    config={'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
                )
            },
            {
                'name': 'transformer_full',
                'model': create_transformer_model(
                    model_type='full', input_channels=2, sequence_length=2048,
                    config={'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'dropout_rate': 0.1}
                )
            },
            {
                'name': 'transformer_lightweight',
                'model': create_transformer_model(
                    model_type='lightweight', input_channels=2, sequence_length=2048,
                    config={'d_model': 128, 'n_heads': 4, 'n_layers': 3, 'dropout_rate': 0.1}
                )
            }
        ]
        
        # Train and evaluate individual models
        individual_results = []
        
        for model_config in models_to_evaluate:
            try:
                result = self.train_and_evaluate_model(
                    model=model_config['model'],
                    model_name=model_config['name'],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config
                )
                individual_results.append(result)
                
                # Store in class attribute for ensemble creation
                self.model_results[model_config['name']] = result
                
            except Exception as e:
                print(f"Error training {model_config['name']}: {e}")
                continue
        
        # Create and evaluate ensembles
        ensemble_results = {}
        if len(individual_results) >= 2:
            try:
                ensemble_results = self.create_and_evaluate_ensembles(
                    individual_results, val_loader, test_loader
                )
                self.ensemble_results = ensemble_results
            except Exception as e:
                print(f"Error creating ensembles: {e}")
        
        # Create comprehensive comparison
        comparison_df, insights = self.create_comprehensive_comparison(
            individual_results, ensemble_results
        )
        
        # Generate final report
        final_report = self.generate_final_report(
            individual_results, ensemble_results, insights, config
        )
        
        # Save final report
        with open(self.results_dir / 'final_evaluation_report.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            json.dump(final_report, f, indent=2, default=convert_numpy)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {self.results_dir}")
        print(f"Best model: {insights['summary']['best_model']}")
        print(f"Best F1 score: {insights['summary']['best_overall_f1']:.4f}")
        
        if ensemble_results:
            best_ensemble = insights['ensemble_analysis']['best_ensemble_method']
            improvement = insights['ensemble_analysis']['improvement_over_individual']
            print(f"Best ensemble: {best_ensemble} (+{improvement:.1f}% improvement)")
        
        return final_report
    
    def generate_final_report(
        self,
        individual_results: List[Dict[str, Any]],
        ensemble_results: Dict[str, Any],
        insights: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive final evaluation report."""
        report = {
            'evaluation_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'configuration': config,
                'total_models_evaluated': len(individual_results),
                'ensemble_methods_tested': list(ensemble_results.keys()) if ensemble_results else []
            },
            'performance_summary': insights['summary'],
            'individual_model_results': {
                result['model_name']: {
                    'test_f1': result['test_optimal_metrics']['f1_score'],
                    'test_precision': result['test_optimal_metrics']['precision'],
                    'test_recall': result['test_optimal_metrics']['recall'],
                    'test_roc_auc': result['test_optimal_metrics']['roc_auc'],
                    'parameters': result.get('model_info', {}).get('total_parameters', 0),
                    'training_time': result['training_time'],
                    'efficiency_metrics': result.get('efficiency_metrics', {})
                }
                for result in individual_results
            },
            'ensemble_results': {
                method: {
                    'test_f1': result['test_optimal_metrics']['f1_score'],
                    'test_precision': result['test_optimal_metrics']['precision'],
                    'test_recall': result['test_optimal_metrics']['recall'],
                    'test_roc_auc': result['test_optimal_metrics']['roc_auc'],
                    'uncertainty_metrics': result.get('test_uncertainty', {}),
                    'ensemble_info': result.get('ensemble_info', {})
                }
                for method, result in ensemble_results.items()
            } if ensemble_results else {},
            'key_insights': insights,
            'recommendations': insights.get('recommendations', []),
            'files_generated': {
                'individual_models': [f"individual_models/{result['model_name']}/" for result in individual_results],
                'ensembles': [f"ensembles/{method}/" for method in ensemble_results.keys()] if ensemble_results else [],
                'comparison_plots': [
                    'comprehensive_comparison/comprehensive_performance_analysis.png',
                    'comprehensive_comparison/efficiency_analysis.png',
                    'comprehensive_comparison/uncertainty_analysis.png'
                ],
                'data_files': [
                    'comprehensive_comparison/model_comparison.csv',
                    'comprehensive_comparison/performance_insights.json'
                ]
            }
        }
        
        return report


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of exoplanet detection models')
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--results-dir', type=str, default='results/advanced_models', 
                       help='Directory to save results')
    parser.add_argument('--create-mock', action='store_true', help='Create mock dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        results_dir=Path(args.results_dir),
        device=args.device
    )
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': 5,
        'val_split': 0.2,
        'test_split': 0.1,
        'synthetic_ratio': 0.3
    }
    
    # Determine dataset path
    dataset_path = None
    if not args.create_mock and args.data_path:
        dataset_path = Path(args.data_path)
    
    # Run comprehensive evaluation
    final_report = evaluator.run_comprehensive_evaluation(
        dataset_path=dataset_path,
        config=config
    )
    
    print(f"\nComprehensive evaluation completed successfully!")
    print(f"Final report saved to: {args.results_dir}/final_evaluation_report.json")


if __name__ == '__main__':
    main()