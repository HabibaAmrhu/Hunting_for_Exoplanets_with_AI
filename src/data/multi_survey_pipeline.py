"""
Advanced data pipeline with multi-survey integration for exoplanet detection.
Supports TESS, Kepler, K2, and other survey data with unified processing.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

from .dataset import ExoplanetDataset
from .types import LightCurveData, SurveyMetadata


@dataclass
class SurveyConfig:
    """
    Configuration for survey data processing.
    """
    name: str
    cadence: float  # seconds
    duration: float  # days
    magnitude_limit: float
    noise_level: float
    bandpass: str
    pixel_scale: float  # arcsec/pixel
    field_of_view: float  # degrees
    
    def __post_init__(self):
        """Validate configuration."""
        if self.cadence <= 0:
            raise ValueError("Cadence must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")


class SurveyDataProcessor(ABC):
    """
    Abstract base class for survey-specific data processors.
    """
    
    def __init__(self, config: SurveyConfig):
        """
        Initialize survey processor.
        
        Args:
            config: Survey configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    @abstractmethod
    def process_light_curve(
        self,
        raw_data: Dict[str, Any]
    ) -> LightCurveData:
        """
        Process raw light curve data.
        
        Args:
            raw_data: Raw data from survey
            
        Returns:
            Processed light curve data
        """
        pass
    
    @abstractmethod
    def validate_data_quality(
        self,
        light_curve: LightCurveData
    ) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Args:
            light_curve: Light curve data
            
        Returns:
            Tuple of (is_valid, quality_issues)
        """
        pass
    
    def normalize_cadence(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        target_cadence: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize cadence to target value.
        
        Args:
            time: Time array
            flux: Flux array
            target_cadence: Target cadence in seconds
            
        Returns:
            Tuple of (normalized_time, normalized_flux)
        """
        if target_cadence is None:
            target_cadence = self.config.cadence
        
        # Calculate current cadence
        current_cadence = np.median(np.diff(time)) * 24 * 3600  # Convert to seconds
        
        if abs(current_cadence - target_cadence) / target_cadence < 0.1:
            return time, flux  # Already close enough
        
        # Interpolate to target cadence
        time_seconds = (time - time[0]) * 24 * 3600
        target_time_seconds = np.arange(
            0, time_seconds[-1], target_cadence
        )
        
        # Linear interpolation
        normalized_flux = np.interp(target_time_seconds, time_seconds, flux)
        normalized_time = time[0] + target_time_seconds / (24 * 3600)
        
        return normalized_time, normalized_flux


class TESSProcessor(SurveyDataProcessor):
    """
    TESS-specific data processor.
    """
    
    def __init__(self):
        config = SurveyConfig(
            name="TESS",
            cadence=120.0,  # 2 minutes
            duration=27.4,  # days per sector
            magnitude_limit=16.0,
            noise_level=60e-6,  # ppm
            bandpass="TESS",
            pixel_scale=21.0,  # arcsec
            field_of_view=24.0  # degrees
        )
        super().__init__(config)
    
    def process_light_curve(
        self,
        raw_data: Dict[str, Any]
    ) -> LightCurveData:
        """
        Process TESS light curve data.
        
        Args:
            raw_data: Raw TESS data
            
        Returns:
            Processed light curve data
        """
        # Extract TESS-specific fields
        time = raw_data.get('TIME', np.array([]))
        flux = raw_data.get('PDCSAP_FLUX', raw_data.get('SAP_FLUX', np.array([])))
        flux_err = raw_data.get('PDCSAP_FLUX_ERR', raw_data.get('SAP_FLUX_ERR', np.array([])))
        quality = raw_data.get('QUALITY', np.zeros_like(time))
        
        # Remove invalid data points
        valid_mask = (
            np.isfinite(time) & 
            np.isfinite(flux) & 
            np.isfinite(flux_err) &
            (quality == 0)  # Good quality data
        )
        
        time = time[valid_mask]
        flux = flux[valid_mask]
        flux_err = flux_err[valid_mask]
        
        # Normalize flux
        median_flux = np.median(flux)
        flux = flux / median_flux
        flux_err = flux_err / median_flux
        
        # Create metadata
        metadata = SurveyMetadata(
            survey="TESS",
            sector=raw_data.get('SECTOR', 0),
            camera=raw_data.get('CAMERA', 0),
            ccd=raw_data.get('CCD', 0),
            magnitude=raw_data.get('TESSMAG', np.nan),
            coordinates=raw_data.get('coordinates', None),
            processing_version=raw_data.get('DATA_REL', 'unknown')
        )
        
        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            metadata=metadata
        )
    
    def validate_data_quality(
        self,
        light_curve: LightCurveData
    ) -> Tuple[bool, List[str]]:
        """
        Validate TESS data quality.
        
        Args:
            light_curve: Light curve data
            
        Returns:
            Tuple of (is_valid, quality_issues)
        """
        issues = []
        
        # Check minimum number of points
        if len(light_curve.time) < 1000:
            issues.append(f"Insufficient data points: {len(light_curve.time)}")
        
        # Check time coverage
        duration = (light_curve.time[-1] - light_curve.time[0])
        if duration < 20.0:  # Less than 20 days
            issues.append(f"Insufficient time coverage: {duration:.1f} days")
        
        # Check flux variability
        flux_std = np.std(light_curve.flux)
        if flux_std > 0.1:  # More than 10% variability
            issues.append(f"High flux variability: {flux_std:.3f}")
        
        # Check for gaps
        time_diffs = np.diff(light_curve.time)
        large_gaps = np.sum(time_diffs > 1.0)  # Gaps > 1 day
        if large_gaps > 5:
            issues.append(f"Too many large gaps: {large_gaps}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class KeplerProcessor(SurveyDataProcessor):
    """
    Kepler/K2-specific data processor.
    """
    
    def __init__(self, mission: str = "Kepler"):
        if mission == "Kepler":
            config = SurveyConfig(
                name="Kepler",
                cadence=1765.5,  # ~29.4 minutes
                duration=90.0,  # days per quarter
                magnitude_limit=17.0,
                noise_level=20e-6,  # ppm
                bandpass="Kepler",
                pixel_scale=3.98,  # arcsec
                field_of_view=105.0  # square degrees
            )
        else:  # K2
            config = SurveyConfig(
                name="K2",
                cadence=1765.5,  # ~29.4 minutes
                duration=80.0,  # days per campaign
                magnitude_limit=17.0,
                noise_level=30e-6,  # ppm
                bandpass="Kepler",
                pixel_scale=3.98,  # arcsec
                field_of_view=105.0  # square degrees
            )
        
        super().__init__(config)
        self.mission = mission
    
    def process_light_curve(
        self,
        raw_data: Dict[str, Any]
    ) -> LightCurveData:
        """
        Process Kepler/K2 light curve data.
        
        Args:
            raw_data: Raw Kepler/K2 data
            
        Returns:
            Processed light curve data
        """
        # Extract Kepler-specific fields
        time = raw_data.get('TIME', np.array([]))
        flux = raw_data.get('PDCSAP_FLUX', raw_data.get('SAP_FLUX', np.array([])))
        flux_err = raw_data.get('PDCSAP_FLUX_ERR', raw_data.get('SAP_FLUX_ERR', np.array([])))
        quality = raw_data.get('SAP_QUALITY', np.zeros_like(time))
        
        # Remove invalid data points
        valid_mask = (
            np.isfinite(time) & 
            np.isfinite(flux) & 
            np.isfinite(flux_err) &
            (quality == 0)  # Good quality data
        )
        
        time = time[valid_mask]
        flux = flux[valid_mask]
        flux_err = flux_err[valid_mask]
        
        # Normalize flux
        median_flux = np.median(flux)
        flux = flux / median_flux
        flux_err = flux_err / median_flux
        
        # Create metadata
        metadata = SurveyMetadata(
            survey=self.mission,
            sector=raw_data.get('QUARTER', raw_data.get('CAMPAIGN', 0)),
            camera=raw_data.get('CHANNEL', 0),
            ccd=0,
            magnitude=raw_data.get('KEPMAG', np.nan),
            coordinates=raw_data.get('coordinates', None),
            processing_version=raw_data.get('DATA_REL', 'unknown')
        )
        
        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            metadata=metadata
        )
    
    def validate_data_quality(
        self,
        light_curve: LightCurveData
    ) -> Tuple[bool, List[str]]:
        """
        Validate Kepler/K2 data quality.
        
        Args:
            light_curve: Light curve data
            
        Returns:
            Tuple of (is_valid, quality_issues)
        """
        issues = []
        
        # Check minimum number of points
        if len(light_curve.time) < 2000:
            issues.append(f"Insufficient data points: {len(light_curve.time)}")
        
        # Check time coverage
        duration = (light_curve.time[-1] - light_curve.time[0])
        if duration < 60.0:  # Less than 60 days
            issues.append(f"Insufficient time coverage: {duration:.1f} days")
        
        # Check flux precision
        flux_precision = np.median(light_curve.flux_err)
        if flux_precision > 0.001:  # Worse than 0.1%
            issues.append(f"Poor flux precision: {flux_precision:.4f}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class MultiSurveyDataPipeline:
    """
    Unified data pipeline for multiple surveys.
    """
    
    def __init__(
        self,
        processors: Optional[Dict[str, SurveyDataProcessor]] = None,
        cache_dir: Optional[Path] = None,
        max_workers: int = 4
    ):
        """
        Initialize multi-survey pipeline.
        
        Args:
            processors: Dictionary of survey processors
            cache_dir: Directory for caching processed data
            max_workers: Maximum number of worker threads
        """
        self.processors = processors or {
            'TESS': TESSProcessor(),
            'Kepler': KeplerProcessor('Kepler'),
            'K2': KeplerProcessor('K2')
        }
        
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_survey': {survey: {'processed': 0, 'successful': 0, 'failed': 0} 
                         for survey in self.processors.keys()}
        }
    
    def process_light_curves(
        self,
        data_sources: List[Dict[str, Any]],
        target_cadence: Optional[float] = None,
        quality_filter: bool = True
    ) -> List[LightCurveData]:
        """
        Process multiple light curves from different surveys.
        
        Args:
            data_sources: List of data source dictionaries
            target_cadence: Target cadence for normalization
            quality_filter: Whether to apply quality filtering
            
        Returns:
            List of processed light curve data
        """
        processed_curves = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit processing tasks
            future_to_source = {
                executor.submit(
                    self._process_single_curve,
                    source,
                    target_cadence,
                    quality_filter
                ): source for source in data_sources
            }
            
            # Collect results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_curves.append(result)
                        self._update_stats(source['survey'], success=True)
                    else:
                        self._update_stats(source['survey'], success=False)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {source}: {e}")
                    self._update_stats(source['survey'], success=False)
        
        return processed_curves
    
    def _process_single_curve(
        self,
        source: Dict[str, Any],
        target_cadence: Optional[float],
        quality_filter: bool
    ) -> Optional[LightCurveData]:
        """
        Process a single light curve.
        
        Args:
            source: Data source information
            target_cadence: Target cadence for normalization
            quality_filter: Whether to apply quality filtering
            
        Returns:
            Processed light curve data or None if failed
        """
        survey = source.get('survey')
        if survey not in self.processors:
            self.logger.warning(f"No processor for survey: {survey}")
            return None
        
        processor = self.processors[survey]
        
        try:
            # Check cache first
            if self.cache_dir:
                cache_key = self._generate_cache_key(source)
                cached_result = self._load_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Process light curve
            light_curve = processor.process_light_curve(source['data'])
            
            # Normalize cadence if requested
            if target_cadence is not None:
                time_norm, flux_norm = processor.normalize_cadence(
                    light_curve.time, light_curve.flux, target_cadence
                )
                light_curve.time = time_norm
                light_curve.flux = flux_norm
                
                # Also normalize flux errors
                if light_curve.flux_err is not None:
                    _, flux_err_norm = processor.normalize_cadence(
                        light_curve.time, light_curve.flux_err, target_cadence
                    )
                    light_curve.flux_err = flux_err_norm
            
            # Quality filtering
            if quality_filter:
                is_valid, issues = processor.validate_data_quality(light_curve)
                if not is_valid:
                    self.logger.info(f"Quality issues for {survey} data: {issues}")
                    return None
            
            # Cache result
            if self.cache_dir:
                self._save_to_cache(cache_key, light_curve)
            
            return light_curve
            
        except Exception as e:
            self.logger.error(f"Error processing {survey} light curve: {e}")
            return None
    
    def create_unified_dataset(
        self,
        light_curves: List[LightCurveData],
        sequence_length: int = 2048,
        augment: bool = True
    ) -> ExoplanetDataset:
        """
        Create unified dataset from multi-survey light curves.
        
        Args:
            light_curves: List of processed light curves
            sequence_length: Target sequence length
            augment: Whether to apply data augmentation
            
        Returns:
            Unified exoplanet dataset
        """
        # Group by survey for balanced sampling
        survey_groups = {}
        for lc in light_curves:
            survey = lc.metadata.survey
            if survey not in survey_groups:
                survey_groups[survey] = []
            survey_groups[survey].append(lc)
        
        # Log survey distribution
        for survey, curves in survey_groups.items():
            self.logger.info(f"{survey}: {len(curves)} light curves")
        
        # Create dataset with survey-aware sampling
        dataset = ExoplanetDataset(
            light_curves=light_curves,
            sequence_length=sequence_length,
            augment=augment
        )
        
        # Add survey information to dataset
        dataset.survey_groups = survey_groups
        
        return dataset
    
    def cross_survey_validation(
        self,
        light_curves: List[LightCurveData],
        test_survey: str,
        sequence_length: int = 2048
    ) -> Tuple[ExoplanetDataset, ExoplanetDataset]:
        """
        Create train/test split for cross-survey validation.
        
        Args:
            light_curves: List of light curves
            test_survey: Survey to use for testing
            sequence_length: Target sequence length
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_curves = []
        test_curves = []
        
        for lc in light_curves:
            if lc.metadata.survey == test_survey:
                test_curves.append(lc)
            else:
                train_curves.append(lc)
        
        train_dataset = ExoplanetDataset(
            light_curves=train_curves,
            sequence_length=sequence_length,
            augment=True
        )
        
        test_dataset = ExoplanetDataset(
            light_curves=test_curves,
            sequence_length=sequence_length,
            augment=False
        )
        
        self.logger.info(
            f"Cross-survey validation: {len(train_curves)} train, "
            f"{len(test_curves)} test ({test_survey})"
        )
        
        return train_dataset, test_dataset
    
    def _generate_cache_key(self, source: Dict[str, Any]) -> str:
        """Generate cache key for data source."""
        import hashlib
        
        # Create hash from source information
        source_str = json.dumps(source, sort_keys=True, default=str)
        return hashlib.md5(source_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[LightCurveData]:
        """Load processed data from cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.debug(f"Cache load failed: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, light_curve: LightCurveData):
        """Save processed data to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(light_curve, f)
        except Exception as e:
            self.logger.debug(f"Cache save failed: {e}")
    
    def _update_stats(self, survey: str, success: bool):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['successful'] += 1
            self.processing_stats['by_survey'][survey]['successful'] += 1
        else:
            self.processing_stats['failed'] += 1
            self.processing_stats['by_survey'][survey]['failed'] += 1
        
        self.processing_stats['by_survey'][survey]['processed'] += 1
    
    def get_processing_report(self) -> Dict[str, Any]:
        """
        Get processing statistics report.
        
        Returns:
            Processing statistics
        """
        total = self.processing_stats['total_processed']
        success_rate = (
            self.processing_stats['successful'] / total 
            if total > 0 else 0
        )
        
        report = {
            'total_processed': total,
            'success_rate': success_rate,
            'by_survey': {}
        }
        
        for survey, stats in self.processing_stats['by_survey'].items():
            survey_total = stats['processed']
            survey_success_rate = (
                stats['successful'] / survey_total 
                if survey_total > 0 else 0
            )
            
            report['by_survey'][survey] = {
                'processed': survey_total,
                'success_rate': survey_success_rate,
                'successful': stats['successful'],
                'failed': stats['failed']
            }
        
        return report


def create_multi_survey_pipeline(
    surveys: List[str] = None,
    cache_dir: Optional[Path] = None,
    max_workers: int = 4
) -> MultiSurveyDataPipeline:
    """
    Factory function to create multi-survey pipeline.
    
    Args:
        surveys: List of surveys to support
        cache_dir: Directory for caching
        max_workers: Maximum worker threads
        
    Returns:
        Configured multi-survey pipeline
    """
    if surveys is None:
        surveys = ['TESS', 'Kepler', 'K2']
    
    processors = {}
    for survey in surveys:
        if survey == 'TESS':
            processors[survey] = TESSProcessor()
        elif survey == 'Kepler':
            processors[survey] = KeplerProcessor('Kepler')
        elif survey == 'K2':
            processors[survey] = KeplerProcessor('K2')
    
    return MultiSurveyDataPipeline(
        processors=processors,
        cache_dir=cache_dir,
        max_workers=max_workers
    )