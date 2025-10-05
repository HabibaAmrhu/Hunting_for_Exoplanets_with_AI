"""
Comprehensive unit tests for data processing components.
Tests dataset classes, augmentation, preprocessing, and synthetic transit generation.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset import LightCurveDataset, AugmentedLightCurveDataset, collate_fn
from data.types import LightCurve, ProcessedLightCurve, PreprocessingConfig
from data.augmentation import (
    TimeJitter, AmplitudeScaling, GaussianNoise, RandomMasking,
    FrequencyFiltering, TimeWarping, MixUp, TransitPreservingAugmentation,
    create_standard_augmentation_pipeline, create_conservative_augmentation_pipeline
)
from preprocessing.preprocessor import LightCurvePreprocessor
from preprocessing.synthetic_injection import SyntheticTransitInjector
from preprocessing.mandel_agol import MandelAgolTransitModel, TransitParams


class TestLightCurveDataset:
    """Test suite for LightCurveDataset."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_samples = 10
        self.sequence_length = 2048
        self.n_channels = 2
        
        # Create mock data
        self.data = np.random.randn(self.n_samples, self.n_channels, self.sequence_length)
        self.labels = np.random.randint(0, 2, self.n_samples)
        self.metadata = [{'star_id': f'test_{i}'} for i in range(self.n_samples)]
        
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = LightCurveDataset(self.data, self.labels, self.metadata)
        
        assert len(dataset) == self.n_samples
        assert dataset.data.shape == self.data.shape
        assert len(dataset.labels) == self.n_samples
        assert len(dataset.metadata) == self.n_samples
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = LightCurveDataset(self.data, self.labels, self.metadata)
        
        data, label, metadata = dataset[0]
        
        assert isinstance(data, torch.Tensor)
        assert data.shape == (self.n_channels, self.sequence_length)
        assert isinstance(label, (int, float, torch.Tensor))
        assert isinstance(metadata, dict)
        assert 'star_id' in metadata
        
    def test_dataset_slice(self):
        """Test dataset slicing."""
        dataset = LightCurveDataset(self.data, self.labels, self.metadata)
        
        subset = dataset[2:5]
        
        assert len(subset) == 3
        
    def test_collate_function(self):
        """Test custom collate function."""
        dataset = LightCurveDataset(self.data, self.labels, self.metadata)
        
        batch = [dataset[i] for i in range(3)]
        data_batch, labels_batch, metadata_batch = collate_fn(batch)
        
        assert data_batch.shape == (3, self.n_channels, self.sequence_length)
        assert labels_batch.shape == (3,)
        assert len(metadata_batch) == 3


class TestAugmentedDataset:
    """Test suite for AugmentedLightCurveDataset."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_samples = 10
        self.sequence_length = 2048
        self.n_channels = 2
        
        self.data = np.random.randn(self.n_samples, self.n_channels, self.sequence_length)
        self.labels = np.random.randint(0, 2, self.n_samples)
        self.metadata = [{'star_id': f'test_{i}'} for i in range(self.n_samples)]
        
    def test_augmented_dataset_initialization(self):
        """Test augmented dataset initialization."""
        augmentation = TimeJitter(max_shift=10, probability=0.5)
        
        dataset = AugmentedLightCurveDataset(
            self.data, self.labels, self.metadata,
            augmentation_pipeline=augmentation
        )
        
        assert len(dataset) == self.n_samples
        assert dataset.augmentation_pipeline is not None
        
    def test_augmentation_application(self):
        """Test that augmentation is applied."""
        augmentation = GaussianNoise(noise_std=0.01, probability=1.0)  # Always apply
        
        dataset = AugmentedLightCurveDataset(
            self.data, self.labels, self.metadata,
            augmentation_pipeline=augmentation
        )
        
        original_data = self.data[0]
        augmented_data, _, _ = dataset[0]
        
        # Data should be different due to noise
        assert not np.allclose(original_data, augmented_data.numpy(), atol=1e-6)


class TestAugmentationTechniques:
    """Test suite for augmentation techniques."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sequence_length = 2048
        self.n_channels = 2
        self.sample_data = np.random.randn(self.n_channels, self.sequence_length)
        self.sample_metadata = {'star_id': 'test_star'}
        
    def test_time_jitter(self):
        """Test TimeJitter augmentation."""
        augmentation = TimeJitter(max_shift=50, probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'time_shift' in augmented_metadata
        
    def test_amplitude_scaling(self):
        """Test AmplitudeScaling augmentation."""
        augmentation = AmplitudeScaling(scale_range=(0.8, 1.2), probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'amplitude_scale' in augmented_metadata
        
        # Check that scaling was applied
        scale_factor = augmented_metadata['amplitude_scale']
        assert 0.8 <= scale_factor <= 1.2
        
    def test_gaussian_noise(self):
        """Test GaussianNoise augmentation."""
        augmentation = GaussianNoise(noise_std=0.01, probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'noise_std' in augmented_metadata
        
        # Data should be different due to noise
        assert not np.allclose(self.sample_data, augmented_data, atol=1e-6)
        
    def test_random_masking(self):
        """Test RandomMasking augmentation."""
        augmentation = RandomMasking(mask_fraction=0.1, probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'masked_points' in augmented_metadata
        
    def test_frequency_filtering(self):
        """Test FrequencyFiltering augmentation."""
        augmentation = FrequencyFiltering(probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'filter_applied' in augmented_metadata
        
    def test_time_warping(self):
        """Test TimeWarping augmentation."""
        augmentation = TimeWarping(warp_strength=0.1, probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'warp_strength' in augmented_metadata
        
    def test_mixup(self):
        """Test MixUp augmentation."""
        # Create second sample for mixing
        other_data = np.random.randn(self.n_channels, self.sequence_length)
        other_metadata = {'star_id': 'other_star'}
        
        augmentation = MixUp(alpha=0.2, probability=1.0)
        augmentation.set_dataset_samples([(other_data, 0, other_metadata)])
        
        augmented_data, augmented_metadata = augmentation(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape
        assert 'mixup_lambda' in augmented_metadata
        assert 'mixed_with' in augmented_metadata
        
    def test_transit_preserving_augmentation(self):
        """Test TransitPreservingAugmentation."""
        # Create data with a mock transit
        transit_data = self.sample_data.copy()
        transit_data[:, 1000:1100] -= 0.01  # Add transit-like dip
        
        augmentation = TransitPreservingAugmentation(probability=1.0)
        
        augmented_data, augmented_metadata = augmentation(
            transit_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == transit_data.shape
        assert 'transit_preserved' in augmented_metadata
        
    def test_augmentation_pipelines(self):
        """Test augmentation pipeline creation."""
        # Standard pipeline
        standard_pipeline = create_standard_augmentation_pipeline()
        assert standard_pipeline is not None
        
        # Conservative pipeline
        conservative_pipeline = create_conservative_augmentation_pipeline()
        assert conservative_pipeline is not None
        
        # Test pipeline application
        augmented_data, augmented_metadata = standard_pipeline(
            self.sample_data.copy(), self.sample_metadata.copy()
        )
        
        assert augmented_data.shape == self.sample_data.shape


class TestPreprocessor:
    """Test suite for LightCurvePreprocessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PreprocessingConfig(
            target_length=2048,
            detrend_method='median',
            normalization='zscore'
        )
        self.preprocessor = LightCurvePreprocessor(self.config)
        
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor.config.target_length == 2048
        assert self.preprocessor.config.detrend_method == 'median'
        assert self.preprocessor.config.normalization == 'zscore'
        
    def test_length_standardization(self):
        """Test length standardization."""
        # Test with shorter sequence
        short_flux = np.random.randn(1000)
        standardized = self.preprocessor._standardize_length(short_flux)
        assert len(standardized) == 2048
        
        # Test with longer sequence
        long_flux = np.random.randn(3000)
        standardized = self.preprocessor._standardize_length(long_flux)
        assert len(standardized) == 2048
        
    def test_detrending(self):
        """Test detrending methods."""
        # Create flux with trend
        time = np.linspace(0, 100, 2048)
        flux = 1.0 + 0.001 * time + 0.01 * np.random.randn(2048)
        
        # Test median detrending
        detrended = self.preprocessor._detrend(flux, method='median')
        assert len(detrended) == len(flux)
        assert np.abs(np.median(detrended)) < 0.01  # Should be close to zero
        
        # Test Savitzky-Golay detrending
        detrended_sg = self.preprocessor._detrend(flux, method='savgol')
        assert len(detrended_sg) == len(flux)
        
    def test_normalization(self):
        """Test normalization methods."""
        flux = np.random.randn(2048) * 10 + 5  # Mean=5, std=10
        
        # Test z-score normalization
        normalized_z = self.preprocessor._normalize(flux, method='zscore')
        assert np.abs(np.mean(normalized_z)) < 1e-10
        assert np.abs(np.std(normalized_z) - 1.0) < 1e-10
        
        # Test min-max normalization
        normalized_minmax = self.preprocessor._normalize(flux, method='minmax')
        assert np.min(normalized_minmax) >= -1.0
        assert np.max(normalized_minmax) <= 1.0
        
    def test_full_preprocessing(self):
        """Test full preprocessing pipeline."""
        # Create mock light curve
        time = np.linspace(0, 100, 1500)  # Different length
        flux = 1.0 + 0.001 * time + 0.01 * np.random.randn(1500)
        flux_err = 0.01 * np.ones_like(flux)
        
        light_curve = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            star_id='test_star'
        )
        
        processed = self.preprocessor.process(light_curve)
        
        assert isinstance(processed, ProcessedLightCurve)
        assert len(processed.flux) == 2048
        assert processed.star_id == 'test_star'
        assert processed.preprocessing_info is not None


class TestSyntheticTransitInjection:
    """Test suite for synthetic transit injection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.injector = SyntheticTransitInjector(
            stellar_catalog='kepler',
            planet_survey='kepler',
            noise_model='realistic'
        )
        
    def test_injector_initialization(self):
        """Test injector initialization."""
        assert self.injector.stellar_catalog == 'kepler'
        assert self.injector.planet_survey == 'kepler'
        assert self.injector.noise_model == 'realistic'
        
    def test_parameter_sampling(self):
        """Test parameter sampling."""
        stellar_params = self.injector.sample_stellar_parameters()
        
        assert 'temperature' in stellar_params
        assert 'radius' in stellar_params
        assert 'mass' in stellar_params
        assert 'magnitude' in stellar_params
        
        # Check reasonable ranges
        assert 3000 <= stellar_params['temperature'] <= 8000
        assert 0.1 <= stellar_params['radius'] <= 3.0
        
    def test_transit_parameter_sampling(self):
        """Test transit parameter sampling."""
        stellar_params = {
            'temperature': 5778,
            'radius': 1.0,
            'mass': 1.0,
            'magnitude': 12.0
        }
        
        transit_params = self.injector.sample_transit_parameters(stellar_params)
        
        assert 'period' in transit_params
        assert 'radius_ratio' in transit_params
        assert 'impact_parameter' in transit_params
        assert 'limb_darkening' in transit_params
        
        # Check reasonable ranges
        assert 0.5 <= transit_params['period'] <= 500
        assert 0.0 <= transit_params['impact_parameter'] <= 1.0
        
    def test_synthetic_transit_generation(self):
        """Test synthetic transit generation."""
        # Create base light curve
        time = np.linspace(0, 100, 2048)
        flux = np.ones_like(time) + 0.001 * np.random.randn(len(time))
        
        stellar_params = {
            'temperature': 5778,
            'radius': 1.0,
            'mass': 1.0,
            'magnitude': 12.0
        }
        
        augmented_flux, transit_info = self.injector.inject_transit(
            time, flux, stellar_params
        )
        
        assert len(augmented_flux) == len(flux)
        assert 'transit_params' in transit_info
        assert 'injection_success' in transit_info
        
        # Check that transit was actually injected
        if transit_info['injection_success']:
            assert not np.allclose(flux, augmented_flux)


class TestMandelAgolModel:
    """Test suite for Mandel-Agol transit model."""
    
    def test_transit_params_validation(self):
        """Test TransitParams validation."""
        # Valid parameters
        valid_params = TransitParams(
            period=10.0,
            radius_ratio=0.1,
            impact_parameter=0.5,
            limb_darkening=[0.3, 0.2]
        )
        
        assert valid_params.period == 10.0
        assert valid_params.radius_ratio == 0.1
        
        # Invalid parameters should raise errors
        with pytest.raises(ValueError):
            TransitParams(
                period=-1.0,  # Invalid period
                radius_ratio=0.1,
                impact_parameter=0.5,
                limb_darkening=[0.3, 0.2]
            )
            
    def test_mandel_agol_computation(self):
        """Test Mandel-Agol transit computation."""
        model = MandelAgolTransitModel()
        
        params = TransitParams(
            period=10.0,
            radius_ratio=0.1,
            impact_parameter=0.3,
            limb_darkening=[0.3, 0.2]
        )
        
        time = np.linspace(-0.5, 0.5, 1000)
        
        flux = model.compute_transit(time, params)
        
        assert len(flux) == len(time)
        assert np.all(flux <= 1.0)  # Transit causes dimming
        assert np.min(flux) < 1.0   # Should have transit depth
        
        # Check symmetry (approximately)
        mid_idx = len(flux) // 2
        left_half = flux[:mid_idx]
        right_half = flux[mid_idx:][::-1]  # Reverse right half
        
        # Should be approximately symmetric
        assert np.allclose(left_half, right_half, rtol=0.1)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])