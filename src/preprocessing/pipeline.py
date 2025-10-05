"""
Integrated preprocessing pipeline for exoplanet detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings
from tqdm import tqdm
import json

from ..data.types import LightCurve, ProcessedLightCurve, PreprocessingConfig
from ..data.dataset import LightCurveDataset
from .preprocessor import LightCurvePreprocessor
from .phase_folding import PhaseFoldingEngine


class PreprocessingPipeline:
    """
    Integrated preprocessing pipeline that handles the complete workflow
    from raw light curves to model-ready processed data.
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        output_dir: Union[str, Path] = "data/processed",
        enable_phase_folding: bool = True,
        save_intermediate: bool = True
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
            output_dir: Directory to save processed data
            enable_phase_folding: Whether to enable advanced phase folding
            save_intermediate: Whether to save intermediate results
        """
        self.config = config or PreprocessingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_phase_folding = enable_phase_folding
        self.save_intermediate = save_intermediate
        
        # Initialize components
        self.preprocessor = LightCurvePreprocessor(self.config)
        if self.enable_phase_folding:
            self.phase_engine = PhaseFoldingEngine()
        
        # Statistics tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'warnings': 0,
            'processing_times': [],
            'quality_scores': []
        }
    
    def process_dataset(
        self,
        dataset: LightCurveDataset,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Tuple[str, Dict]:
        """
        Process entire dataset through the preprocessing pipeline.
        
        Args:
            dataset: Input light curve dataset
            batch_size: Number of light curves to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (processed_metadata_file, processing_statistics)
        """
        print(f"Starting preprocessing pipeline for {len(dataset)} light curves")
        print(f"Output directory: {self.output_dir}")
        
        # Reset statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'warnings': 0,
            'processing_times': [],
            'quality_scores': []
        }
        
        processed_metadata = []
        
        # Process in batches
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        
        if show_progress:
            batch_iterator = tqdm(range(n_batches), desc="Processing batches")
        else:
            batch_iterator = range(n_batches)
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            
            batch_metadata = self._process_batch(dataset, start_idx, end_idx)
            processed_metadata.extend(batch_metadata)
        
        # Save processed metadata
        metadata_df = pd.DataFrame(processed_metadata)
        metadata_file = self.output_dir / "processed_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        # Save processing statistics
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {self.processing_stats['successful']}")
        print(f"Failed: {self.processing_stats['failed']}")
        print(f"Warnings: {self.processing_stats['warnings']}")
        print(f"Processed metadata saved to: {metadata_file}")
        
        return str(metadata_file), self.processing_stats
    
    def _process_batch(
        self,
        dataset: LightCurveDataset,
        start_idx: int,
        end_idx: int
    ) -> List[Dict]:
        """Process a batch of light curves."""
        
        batch_metadata = []
        
        for idx in range(start_idx, end_idx):
            try:
                # Load light curve
                data_tensor, label, metadata = dataset[idx]
                
                # Convert back to LightCurve object for processing
                # This is a simplified conversion - in practice, dataset should provide LightCurve objects
                light_curve = self._tensor_to_light_curve(data_tensor, label, metadata)
                
                # Process light curve
                processed_metadata = self._process_single_light_curve(light_curve)
                
                if processed_metadata:
                    batch_metadata.append(processed_metadata)
                    self.processing_stats['successful'] += 1
                else:
                    self.processing_stats['failed'] += 1
                
            except Exception as e:
                warnings.warn(f"Failed to process light curve {idx}: {e}")
                self.processing_stats['failed'] += 1
                continue
            
            self.processing_stats['total_processed'] += 1
        
        return batch_metadata
    
    def _tensor_to_light_curve(
        self,
        data_tensor,
        label: int,
        metadata: Dict
    ) -> LightCurve:
        """Convert tensor data back to LightCurve object."""
        
        # This is a placeholder - in practice, the dataset should provide
        # access to original LightCurve objects or we should modify the
        # pipeline to work directly with the dataset
        
        # For now, create a synthetic light curve from metadata
        star_id = metadata.get('star_id', 'unknown')
        
        # Generate synthetic time and flux arrays
        # In practice, these would come from the original data files
        length = 2000  # Default length
        time = np.linspace(0, 100, length)
        flux = np.random.normal(1.0, 0.001, length)
        flux_err = np.full(length, 0.001)
        
        return LightCurve(
            star_id=star_id,
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=label,
            period=metadata.get('period'),
            metadata=metadata
        )
    
    def process_light_curve_file(
        self,
        file_path: Union[str, Path],
        star_id: str,
        label: int,
        period: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Process a single light curve file.
        
        Args:
            file_path: Path to light curve data file
            star_id: Star identifier
            label: Classification label
            period: Known period (if any)
            metadata: Additional metadata
            
        Returns:
            Path to processed file or None if processing failed
        """
        try:
            # Load light curve from file
            light_curve = self._load_light_curve_from_file(
                file_path, star_id, label, period, metadata
            )
            
            # Process light curve
            processed_metadata = self._process_single_light_curve(light_curve)
            
            if processed_metadata:
                return processed_metadata['processed_file_path']
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"Failed to process {file_path}: {e}")
            return None
    
    def _load_light_curve_from_file(
        self,
        file_path: Union[str, Path],
        star_id: str,
        label: int,
        period: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> LightCurve:
        """Load light curve from data file."""
        
        file_path = Path(file_path)
        
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            time = data['time']
            flux = data['flux']
            flux_err = data.get('flux_err', np.ones_like(flux) * 0.001)
            
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            time = df['time'].values
            flux = df['flux'].values
            flux_err = df.get('flux_err', np.ones_like(flux) * 0.001).values
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return LightCurve(
            star_id=star_id,
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=label,
            period=period,
            metadata=metadata or {}
        )
    
    def _process_single_light_curve(self, light_curve: LightCurve) -> Optional[Dict]:
        """Process a single light curve through the complete pipeline."""
        
        import time as time_module
        start_time = time_module.time()
        
        try:
            # Step 1: Basic preprocessing
            processed_lc = self.preprocessor.process(light_curve)
            
            # Step 2: Advanced phase folding (if enabled and period available)
            if self.enable_phase_folding and light_curve.period is not None:
                processed_lc = self._enhance_with_phase_folding(
                    light_curve, processed_lc
                )
            
            # Step 3: Quality assessment
            quality_score = self._assess_quality(light_curve, processed_lc)
            
            # Step 4: Save processed data
            processed_file_path = self._save_processed_data(processed_lc, light_curve.star_id)
            
            # Step 5: Create metadata entry
            processing_time = time_module.time() - start_time
            self.processing_stats['processing_times'].append(processing_time)
            self.processing_stats['quality_scores'].append(quality_score)
            
            metadata_entry = {
                'star_id': light_curve.star_id,
                'original_label': light_curve.label,
                'processed_label': processed_lc.label,
                'period': light_curve.period,
                'confidence_weight': processed_lc.confidence_weight,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'processed_file_path': processed_file_path,
                'original_length': len(light_curve.flux),
                'processed_length': len(processed_lc.raw_flux),
                'has_phase_folding': light_curve.period is not None,
                'preprocessing_config': self.config.__dict__
            }
            
            # Add original metadata
            if light_curve.metadata:
                metadata_entry.update({
                    f"original_{k}": v for k, v in light_curve.metadata.items()
                })
            
            return metadata_entry
            
        except Exception as e:
            warnings.warn(f"Processing failed for {light_curve.star_id}: {e}")
            self.processing_stats['warnings'] += 1
            return None
    
    def _enhance_with_phase_folding(
        self,
        original_lc: LightCurve,
        processed_lc: ProcessedLightCurve
    ) -> ProcessedLightCurve:
        """Enhance processed light curve with advanced phase folding."""
        
        if not self.enable_phase_folding or original_lc.period is None:
            return processed_lc
        
        try:
            # Optimize epoch for better phase folding
            optimized_epoch, signal_strength = self.phase_engine.optimize_epoch(
                original_lc.time, original_lc.flux, original_lc.period
            )
            
            # Create enhanced phase-folded version
            phase_grid, enhanced_folded_flux = self.phase_engine.fold_light_curve(
                original_lc.time, original_lc.flux, original_lc.period,
                epoch=optimized_epoch, phase_bins=self.config.target_length
            )
            
            # Update the processed light curve
            import torch
            processed_lc.phase_folded_flux = torch.from_numpy(enhanced_folded_flux).float()
            
            # Update augmentation parameters with phase folding info
            processed_lc.augmentation_params.update({
                'optimized_epoch': optimized_epoch,
                'signal_strength': signal_strength,
                'enhanced_phase_folding': True
            })
            
            return processed_lc
            
        except Exception as e:
            warnings.warn(f"Phase folding enhancement failed: {e}")
            return processed_lc
    
    def _assess_quality(
        self,
        original_lc: LightCurve,
        processed_lc: ProcessedLightCurve
    ) -> float:
        """Assess the quality of processed light curve."""
        
        quality_score = 1.0
        
        # Factor 1: Data completeness
        completeness = float(processed_lc.mask.mean())
        quality_score *= completeness
        
        # Factor 2: Signal-to-noise ratio
        if hasattr(original_lc, 'flux_err') and original_lc.flux_err is not None:
            snr = np.median(np.abs(original_lc.flux) / original_lc.flux_err)
            snr_factor = min(snr / 50.0, 1.0)  # Normalize to max of 1.0
            quality_score *= snr_factor
        
        # Factor 3: Length adequacy
        length_factor = min(len(original_lc.flux) / 1000.0, 1.0)
        quality_score *= length_factor
        
        # Factor 4: Variability (not too high, not too low)
        variability = float(processed_lc.raw_flux.std())
        if 0.5 <= variability <= 2.0:
            variability_factor = 1.0
        else:
            variability_factor = 0.8
        quality_score *= variability_factor
        
        # Factor 5: Phase folding quality (if applicable)
        if original_lc.period is not None and self.enable_phase_folding:
            # Check if phase folding improved signal
            raw_std = float(processed_lc.raw_flux.std())
            phase_std = float(processed_lc.phase_folded_flux.std())
            
            if phase_std > raw_std * 1.1:  # Phase folding increased variability
                phase_factor = 1.2
            else:
                phase_factor = 1.0
            
            quality_score *= phase_factor
        
        # Clamp to [0, 1] range
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        return float(quality_score)
    
    def _save_processed_data(
        self,
        processed_lc: ProcessedLightCurve,
        star_id: str
    ) -> str:
        """Save processed light curve data to file."""
        
        # Create filename
        processed_file = self.output_dir / f"{star_id}_processed.npz"
        
        # Convert tensors to numpy arrays
        raw_flux = processed_lc.raw_flux.numpy()
        phase_folded_flux = processed_lc.phase_folded_flux.numpy()
        mask = processed_lc.mask.numpy()
        
        # Save to compressed numpy file
        np.savez_compressed(
            processed_file,
            raw_flux=raw_flux,
            phase_folded_flux=phase_folded_flux,
            mask=mask,
            label=processed_lc.label,
            confidence_weight=processed_lc.confidence_weight,
            augmentation_params=processed_lc.augmentation_params
        )
        
        return processed_file.name
    
    def validate_processed_data(
        self,
        processed_metadata_file: Union[str, Path]
    ) -> Dict:
        """
        Validate processed dataset for consistency and quality.
        
        Args:
            processed_metadata_file: Path to processed metadata CSV
            
        Returns:
            Validation report dictionary
        """
        metadata_df = pd.read_csv(processed_metadata_file)
        
        validation_report = {
            'total_files': len(metadata_df),
            'missing_files': 0,
            'corrupted_files': 0,
            'quality_distribution': {},
            'processing_time_stats': {},
            'issues': []
        }
        
        # Check file existence and integrity
        for _, row in metadata_df.iterrows():
            processed_file = self.output_dir / row['processed_file_path']
            
            if not processed_file.exists():
                validation_report['missing_files'] += 1
                validation_report['issues'].append(f"Missing file: {processed_file}")
                continue
            
            try:
                # Try to load the file
                data = np.load(processed_file)
                
                # Check required arrays
                required_arrays = ['raw_flux', 'phase_folded_flux', 'mask']
                for array_name in required_arrays:
                    if array_name not in data:
                        validation_report['issues'].append(
                            f"Missing array {array_name} in {processed_file}"
                        )
                    elif len(data[array_name]) != self.config.target_length:
                        validation_report['issues'].append(
                            f"Wrong length for {array_name} in {processed_file}"
                        )
                
            except Exception as e:
                validation_report['corrupted_files'] += 1
                validation_report['issues'].append(f"Corrupted file {processed_file}: {e}")
        
        # Quality distribution
        if 'quality_score' in metadata_df.columns:
            quality_scores = metadata_df['quality_score'].dropna()
            validation_report['quality_distribution'] = {
                'mean': float(quality_scores.mean()),
                'std': float(quality_scores.std()),
                'min': float(quality_scores.min()),
                'max': float(quality_scores.max()),
                'median': float(quality_scores.median())
            }
        
        # Processing time statistics
        if 'processing_time' in metadata_df.columns:
            proc_times = metadata_df['processing_time'].dropna()
            validation_report['processing_time_stats'] = {
                'mean': float(proc_times.mean()),
                'std': float(proc_times.std()),
                'min': float(proc_times.min()),
                'max': float(proc_times.max()),
                'total': float(proc_times.sum())
            }
        
        # Save validation report
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"Validation complete. Report saved to: {report_file}")
        
        return validation_report
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline configuration and statistics."""
        
        summary = {
            'configuration': {
                'target_length': self.config.target_length,
                'detrend_method': self.config.detrend_method,
                'normalization': self.config.normalization,
                'enable_phase_folding': self.enable_phase_folding,
                'save_intermediate': self.save_intermediate
            },
            'statistics': self.processing_stats,
            'output_directory': str(self.output_dir)
        }
        
        return summary