"""
Unit tests for the ultra-high accuracy exoplanet detection system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.advanced_detection import (
    AdvancedPreprocessor, 
    PreprocessingConfig, 
    ProcessedData
)


class TestAdvancedPreprocessor:
    """Test suite for AdvancedPreprocessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = PreprocessingConfig()
        self.preprocessor = AdvancedPreprocessor(self.config)
        
        # Create test data
        np.random.seed(42)
        self.time = np.linspace(0, 30, 1000)  # 30 days, 1000 points
        self.flux = np.ones_like(self.time) + np.random.normal(0, 0.001, len(self.time))
        self.flux_err = np.full_like(self.flux, 0.001)
        
        # Add some realistic features
        # Stellar rotation
        self.flux += 0.005 * np.sin(2 * np.pi * self.time / 12.5)
        
        # Add transit signal
        transit_period = 5.2
        transit_depth = 0.008
        for i in range(6):
            transit_center = i * transit_period + 2.5
            transit_mask = np.abs(self.time - transit_center) < 0.1
            self.flux[transit_mask] *= (1 - transit_depth)
        
        # Add some outliers
        outlier_indices = np.random.choice(len(self.flux), 10, replace=False)
        self.flux[outlier_indices] += np.random.normal(0, 0.02, 10)
    
    def test_input_validation_valid_data(self):
        """Test input validation with valid data"""
        # Should not raise any exceptions
        self.preprocessor._validate_input_data(self.time, self.flux, self.flux_err)
    
    def test_input_validation_invalid_types(self):
        """Test input validation with invalid data types"""
        with pytest.raises(ValueError, match="Time and flux must be numpy arrays"):
            self.preprocessor._validate_input_data(self.time.tolist(), self.flux)
    
    def test_input_validation_mismatched_lengths(self):
        """Test input validation with mismatched array lengths"""
        with pytest.raises(ValueError, match="Time and flux arrays must have same length"):
            self.preprocessor._validate_input_data(self.time[:-10], self.flux)
    
    def test_input_validation_too_short(self):
        """Test input validation with too short arrays"""
        short_time = np.linspace(0, 1, 50)
        short_flux = np.ones_like(short_time)
        
        with pytest.raises(ValueError, match="Light curve too short for reliable analysis"):
            self.preprocessor._validate_input_data(short_time, short_flux)
    
    def test_input_validation_nan_values(self):
        """Test input validation with NaN values"""
        flux_with_nan = self.flux.copy()
        flux_with_nan[100] = np.nan
        
        with pytest.raises(ValueError, match="Time or flux contains NaN or infinite values"):
            self.preprocessor._validate_input_data(self.time, flux_with_nan)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        quality_flags = self.preprocessor._assess_data_quality(self.time, self.flux)
        
        assert len(quality_flags) == len(self.flux)
        assert quality_flags.dtype == int
        assert np.any(quality_flags > 0)  # Should flag some outliers
    
    def test_sigma_clipping(self):
        """Test iterative sigma clipping"""
        # Add extreme outliers
        flux_with_outliers = self.flux.copy()
        flux_with_outliers[100:105] = 2.0  # Extreme outliers
        
        flux_clean = self.preprocessor._sigma_clip_outliers(flux_with_outliers)
        
        # Check that outliers were removed/corrected
        assert np.max(flux_clean) < np.max(flux_with_outliers)
        assert len(flux_clean) == len(flux_with_outliers)
    
    def test_savgol_filtering(self):
        """Test Savitzky-Golay filtering"""
        # Add high-frequency noise
        noisy_flux = self.flux + 0.01 * np.random.normal(0, 1, len(self.flux))
        
        filtered_flux = self.preprocessor._savgol_filter(noisy_flux)
        
        # Check that noise was reduced
        original_std = np.std(noisy_flux)
        filtered_std = np.std(filtered_flux)
        assert filtered_std < original_std
        assert len(filtered_flux) == len(noisy_flux)
    
    def test_biweight_detrending(self):
        """Test biweight detrending"""
        # Add long-term trend
        trend = 0.01 * (self.time / np.max(self.time))
        flux_with_trend = self.flux + trend
        
        detrended_flux = self.preprocessor._biweight_detrend(self.time, flux_with_trend)
        
        # Check that trend was removed
        assert np.abs(np.mean(detrended_flux) - 1.0) < 0.01
        assert len(detrended_flux) == len(flux_with_trend)
    
    def test_polynomial_detrending(self):
        """Test polynomial detrending"""
        # Add polynomial trend
        trend = 0.001 * self.time**2
        flux_with_trend = self.flux + trend
        
        detrended_flux = self.preprocessor._polynomial_detrend(self.time, flux_with_trend)
        
        # Check that trend was removed
        assert np.std(detrended_flux) < np.std(flux_with_trend)
        assert len(detrended_flux) == len(flux_with_trend)
    
    def test_robust_normalization(self):
        """Test robust normalization"""
        # Scale flux to different range
        scaled_flux = self.flux * 1000 + 5000
        
        normalized_flux = self.preprocessor._robust_normalize(scaled_flux)
        
        # Check normalization properties
        assert np.abs(np.median(normalized_flux)) < 0.1  # Should be near zero
        mad = np.median(np.abs(normalized_flux - np.median(normalized_flux)))
        assert np.abs(mad - 1.0) < 0.1  # MAD should be near 1
    
    def test_gap_interpolation(self):
        """Test intelligent gap interpolation"""
        # Create test data with larger gaps
        time_sparse = np.linspace(0, 30, 100)  # Sparser data to create larger gaps
        flux_sparse = np.ones_like(time_sparse) + np.random.normal(0, 0.001, len(time_sparse))
        
        # Remove some points to create a significant gap
        gap_indices = np.arange(40, 50)  # 10-point gap in sparse data
        time_with_gaps = np.delete(time_sparse, gap_indices)
        flux_with_gaps = np.delete(flux_sparse, gap_indices)
        
        time_interp, flux_interp, _ = self.preprocessor._interpolate_gaps(
            time_with_gaps, flux_with_gaps, None
        )
        
        # Check that gaps were filled or at least data integrity is maintained
        assert len(time_interp) >= len(time_with_gaps)
        assert len(flux_interp) >= len(flux_with_gaps)
        assert len(time_interp) == len(flux_interp)
        
        # Check that time array is still sorted
        assert np.all(np.diff(time_interp) > 0)
    
    def test_complete_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline"""
        processed_data = self.preprocessor.preprocess(self.time, self.flux, self.flux_err)
        
        # Check return type
        assert isinstance(processed_data, ProcessedData)
        
        # Check data integrity
        assert len(processed_data.time) > 0
        assert len(processed_data.flux) > 0
        assert len(processed_data.time) == len(processed_data.flux)
        
        # Check processing info
        assert 'original_length' in processed_data.preprocessing_info
        assert 'preprocessing_steps' in processed_data.preprocessing_info
        assert 'quality_metrics' in processed_data.preprocessing_info
        
        # Check that multiple steps were applied
        steps = processed_data.preprocessing_info['preprocessing_steps']
        assert len(steps) >= 5  # Should have at least 5 processing steps
        
        # Check quality metrics
        quality_metrics = processed_data.preprocessing_info['quality_metrics']
        assert 'noise_level' in quality_metrics
        assert 'mean_flux' in quality_metrics
        assert 'flux_range' in quality_metrics
    
    def test_preprocessing_with_different_configs(self):
        """Test preprocessing with different configurations"""
        # Test with different detrending method
        config_poly = PreprocessingConfig(detrend_method='polynomial')
        preprocessor_poly = AdvancedPreprocessor(config_poly)
        
        processed_poly = preprocessor_poly.preprocess(self.time, self.flux)
        
        # Test with different normalization method
        config_standard = PreprocessingConfig(normalize_method='standard')
        preprocessor_standard = AdvancedPreprocessor(config_standard)
        
        processed_standard = preprocessor_standard.preprocess(self.time, self.flux)
        
        # Results should be different but valid
        assert not np.array_equal(processed_poly.flux, processed_standard.flux)
        assert len(processed_poly.flux) == len(processed_standard.flux)
    
    def test_preprocessing_preserves_transit_signals(self):
        """Test that preprocessing preserves transit signals"""
        # Create clean transit signal
        clean_time = np.linspace(0, 10, 500)
        clean_flux = np.ones_like(clean_time)
        
        # Add clear transit
        transit_mask = (clean_time > 4.9) & (clean_time < 5.1)
        clean_flux[transit_mask] *= 0.99  # 1% depth
        
        # Add noise and trends
        noisy_flux = clean_flux + 0.002 * np.random.normal(0, 1, len(clean_flux))
        noisy_flux += 0.005 * np.sin(2 * np.pi * clean_time / 8)  # Stellar rotation
        
        # Process the data
        processed_data = self.preprocessor.preprocess(clean_time, noisy_flux)
        
        # Check that transit is still visible
        processed_transit_region = processed_data.flux[transit_mask]
        processed_baseline = np.median(processed_data.flux[~transit_mask])
        
        # Transit should still be detectable (flux dip)
        transit_depth = processed_baseline - np.median(processed_transit_region)
        assert transit_depth > 0  # Should be a dip
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases"""
        # Test with very short window for Savgol filter
        short_flux = np.ones(10)
        filtered = self.preprocessor._savgol_filter(short_flux)
        assert len(filtered) == len(short_flux)
        
        # Test with constant flux (no variation)
        constant_flux = np.ones(1000)
        normalized = self.preprocessor._robust_normalize(constant_flux)
        assert len(normalized) == len(constant_flux)
        assert not np.any(np.isnan(normalized))
    
    def test_performance_requirements(self):
        """Test that preprocessing meets performance requirements"""
        import time
        
        # Test with larger dataset
        large_time = np.linspace(0, 100, 10000)  # 10k points
        large_flux = np.ones_like(large_time) + np.random.normal(0, 0.001, len(large_time))
        
        start_time = time.time()
        processed_data = self.preprocessor.preprocess(large_time, large_flux)
        processing_time = time.time() - start_time
        
        # Should process 10k points in reasonable time (< 5 seconds)
        assert processing_time < 5.0
        assert len(processed_data.flux) == len(large_flux)


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PreprocessingConfig()
        
        assert config.savgol_window_length == 51
        assert config.savgol_polyorder == 3
        assert config.sigma_clip_sigma == 3.0
        assert config.sigma_clip_maxiters == 5
        assert config.detrend_method == 'biweight'
        assert config.normalize_method == 'robust'
        assert config.interpolate_gaps == True
        assert config.max_gap_size == 10
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = PreprocessingConfig(
            savgol_window_length=101,
            sigma_clip_sigma=2.5,
            detrend_method='polynomial',
            normalize_method='standard'
        )
        
        assert config.savgol_window_length == 101
        assert config.sigma_clip_sigma == 2.5
        assert config.detrend_method == 'polynomial'
        assert config.normalize_method == 'standard'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])