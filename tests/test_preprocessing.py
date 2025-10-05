"""Tests for preprocessing functionality."""

import pytest
import numpy as np
import torch
from src.data.types import LightCurve, PreprocessingConfig
from src.preprocessing.preprocessor import LightCurvePreprocessor


class TestLightCurvePreprocessor:
    """Test LightCurvePreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PreprocessingConfig(
            target_length=1024,  # Smaller for testing
            detrend_method='median',
            normalization='zscore'
        )
        self.preprocessor = LightCurvePreprocessor(self.config)
    
    def create_test_light_curve(
        self, 
        length: int = 2000,
        add_trend: bool = True,
        add_noise: bool = True,
        add_transit: bool = False,
        period: float = None
    ) -> LightCurve:
        """Create a synthetic light curve for testing."""
        
        time = np.linspace(0, 100, length)
        flux = np.ones(length)
        
        # Add trend
        if add_trend:
            trend = 0.001 * (time - 50) ** 2 / 100  # Quadratic trend
            flux += trend
        
        # Add noise
        if add_noise:
            noise = np.random.normal(0, 0.001, length)
            flux += noise
        
        # Add transit
        if add_transit and period:
            transit_depth = 0.01
            transit_duration = 0.1 * period
            
            phases = (time % period) / period
            transit_mask = np.abs(phases - 0.5) < (transit_duration / period / 2)
            flux[transit_mask] -= transit_depth
        
        flux_err = np.full(length, 0.001)
        
        return LightCurve(
            star_id="test_star",
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=1 if add_transit else 0,
            period=period,
            metadata={}
        )
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        # Default config
        preprocessor = LightCurvePreprocessor()
        assert preprocessor.config.target_length == 2048
        
        # Custom config
        custom_config = PreprocessingConfig(target_length=1024)
        preprocessor = LightCurvePreprocessor(custom_config)
        assert preprocessor.config.target_length == 1024
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create light curve with missing values
        lc = self.create_test_light_curve(length=1000)
        
        # Introduce NaN values
        lc.flux[100:110] = np.nan
        lc.flux[500:505] = np.inf
        
        time, flux, flux_err = self.preprocessor._handle_missing_values(
            lc.time, lc.flux, lc.flux_err
        )
        
        # Should have removed NaN/inf values
        assert len(flux) < len(lc.flux)
        assert np.all(np.isfinite(flux))
        assert np.all(np.isfinite(time))
        assert np.all(np.isfinite(flux_err))
    
    def test_detrend_median_filter(self):
        """Test median filter detrending."""
        # Create light curve with strong trend
        lc = self.create_test_light_curve(length=1000, add_trend=True, add_noise=False)
        
        detrended = self.preprocessor._detrend_median_filter(lc.flux)
        
        # Detrended should have less variation than original
        original_range = np.max(lc.flux) - np.min(lc.flux)
        detrended_range = np.max(detrended) - np.min(detrended)
        
        assert detrended_range < original_range
        
        # Should be centered around median of original
        assert abs(np.median(detrended) - np.median(lc.flux)) < 0.01
    
    def test_detrend_savitzky_golay(self):
        """Test Savitzky-Golay detrending."""
        # Create light curve with trend
        lc = self.create_test_light_curve(length=1000, add_trend=True, add_noise=False)
        
        detrended = self.preprocessor._detrend_savitzky_golay(lc.flux)
        
        # Should remove trend
        original_range = np.max(lc.flux) - np.min(lc.flux)
        detrended_range = np.max(detrended) - np.min(detrended)
        
        assert detrended_range < original_range
    
    def test_normalize_zscore(self):
        """Test z-score normalization."""
        flux = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        
        normalized = self.preprocessor._normalize(flux)
        
        # Should have zero median and unit std (approximately)
        assert abs(np.median(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        config = PreprocessingConfig(normalization='minmax')
        preprocessor = LightCurvePreprocessor(config)
        
        flux = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        
        normalized = preprocessor._normalize(flux)
        
        # Should be in range [0, 1]
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
    
    def test_standardize_length_downsample(self):
        """Test downsampling to target length."""
        time = np.linspace(0, 100, 2000)
        flux = np.sin(2 * np.pi * time / 10)  # Sine wave
        
        new_time, new_flux = self.preprocessor._standardize_length(time, flux)
        
        assert len(new_flux) == self.config.target_length
        assert len(new_time) == self.config.target_length
        
        # Should preserve overall shape
        assert np.corrcoef(flux[::2], new_flux[::1])[0, 1] > 0.9
    
    def test_standardize_length_upsample(self):
        """Test upsampling to target length."""
        time = np.linspace(0, 100, 500)  # Shorter than target
        flux = np.sin(2 * np.pi * time / 10)
        
        new_time, new_flux = self.preprocessor._standardize_length(time, flux)
        
        assert len(new_flux) == self.config.target_length
        assert len(new_time) == self.config.target_length
    
    def test_create_phase_folded_with_period(self):
        """Test phase folding with known period."""
        period = 10.0
        lc = self.create_test_light_curve(
            length=1000, 
            add_transit=True, 
            period=period
        )
        
        # Standardize first
        time, flux = self.preprocessor._standardize_length(lc.time, lc.flux)
        
        phase_folded = self.preprocessor._create_phase_folded(time, flux, period)
        
        assert len(phase_folded) == len(flux)
        assert np.all(np.isfinite(phase_folded))
    
    def test_create_phase_folded_no_period(self):
        """Test phase folding without period."""
        lc = self.create_test_light_curve(length=1000)
        
        time, flux = self.preprocessor._standardize_length(lc.time, lc.flux)
        
        phase_folded = self.preprocessor._create_phase_folded(time, flux, None)
        
        # Should return copy of original flux
        assert len(phase_folded) == len(flux)
        np.testing.assert_array_equal(phase_folded, flux)
    
    def test_create_mask(self):
        """Test mask creation."""
        flux = np.array([1.0, np.nan, 1.1, np.inf, 0.9])
        
        mask = self.preprocessor._create_mask(flux)
        
        expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(mask, expected)
    
    def test_calculate_confidence_weight(self):
        """Test confidence weight calculation."""
        # Short, noisy light curve
        lc_low = self.create_test_light_curve(length=200, add_noise=True)
        lc_low.flux_err = np.full_like(lc_low.flux, 0.1)  # High noise
        
        # Long, clean light curve  
        lc_high = self.create_test_light_curve(length=5000, add_noise=False)
        lc_high.flux_err = np.full_like(lc_high.flux, 0.001)  # Low noise
        
        weight_low = self.preprocessor._calculate_confidence_weight(lc_low, lc_low.flux)
        weight_high = self.preprocessor._calculate_confidence_weight(lc_high, lc_high.flux)
        
        # High quality should have higher weight
        assert weight_high > weight_low
        
        # Weights should be in reasonable range
        assert 0.5 <= weight_low <= 2.0
        assert 0.5 <= weight_high <= 2.0
    
    def test_full_processing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test light curve with period
        lc = self.create_test_light_curve(
            length=1500,
            add_trend=True,
            add_noise=True,
            add_transit=True,
            period=15.0
        )
        
        processed = self.preprocessor.process(lc)
        
        # Check output structure
        assert isinstance(processed.raw_flux, torch.Tensor)
        assert isinstance(processed.phase_folded_flux, torch.Tensor)
        assert isinstance(processed.mask, torch.Tensor)
        
        # Check dimensions
        assert processed.raw_flux.shape == (self.config.target_length,)
        assert processed.phase_folded_flux.shape == (self.config.target_length,)
        assert processed.mask.shape == (self.config.target_length,)
        
        # Check label preservation
        assert processed.label == lc.label
        
        # Check confidence weight
        assert 0.5 <= processed.confidence_weight <= 2.0
        
        # Check that phase folding is different from raw (when period exists)
        assert not torch.allclose(processed.raw_flux, processed.phase_folded_flux)
    
    def test_processing_without_period(self):
        """Test processing light curve without known period."""
        lc = self.create_test_light_curve(length=1500, period=None)
        
        processed = self.preprocessor.process(lc)
        
        # Phase folded should be same as raw when no period
        torch.testing.assert_close(processed.raw_flux, processed.phase_folded_flux)
    
    def test_batch_processing(self):
        """Test batch processing of multiple light curves."""
        light_curves = [
            self.create_test_light_curve(length=1000 + i*100, period=10.0 + i)
            for i in range(5)
        ]
        
        processed_curves = self.preprocessor.batch_process(
            light_curves, show_progress=False
        )
        
        assert len(processed_curves) == len(light_curves)
        
        for processed in processed_curves:
            assert processed.raw_flux.shape == (self.config.target_length,)
    
    def test_get_preprocessing_stats(self):
        """Test preprocessing statistics."""
        lc = self.create_test_light_curve(length=1500, period=12.0)
        processed = self.preprocessor.process(lc)
        
        stats = self.preprocessor.get_preprocessing_stats(lc, processed)
        
        # Check required keys
        required_keys = [
            'original_length', 'processed_length', 'original_std', 
            'processed_std', 'confidence_weight', 'has_period', 
            'period_value', 'missing_data_fraction'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # Check values
        assert stats['original_length'] == 1500
        assert stats['processed_length'] == self.config.target_length
        assert stats['has_period'] == True
        assert stats['period_value'] == 12.0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very short light curve
        short_lc = LightCurve(
            star_id="short",
            time=np.array([0, 1, 2]),
            flux=np.array([1.0, 1.1, 0.9]),
            flux_err=np.array([0.01, 0.01, 0.01]),
            label=0
        )
        
        processed = self.preprocessor.process(short_lc)
        assert processed.raw_flux.shape == (self.config.target_length,)
        
        # All NaN light curve should raise error
        nan_lc = LightCurve(
            star_id="nan",
            time=np.array([0, 1, 2]),
            flux=np.array([np.nan, np.nan, np.nan]),
            flux_err=np.array([0.01, 0.01, 0.01]),
            label=0
        )
        
        with pytest.raises(ValueError, match="No finite values"):
            self.preprocessor.process(nan_lc)
    
    def test_different_configurations(self):
        """Test different preprocessing configurations."""
        # Test Savitzky-Golay detrending
        savgol_config = PreprocessingConfig(
            detrend_method='savgol',
            normalization='minmax'
        )
        savgol_preprocessor = LightCurvePreprocessor(savgol_config)
        
        lc = self.create_test_light_curve(length=1000, add_trend=True)
        processed = savgol_preprocessor.process(lc)
        
        # Should be normalized to [0, 1] range approximately
        assert torch.min(processed.raw_flux) >= -0.1  # Allow small numerical errors
        assert torch.max(processed.raw_flux) <= 1.1


class TestPhaseFoldingEngine:
    """Test PhaseFoldingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.preprocessing.phase_folding import PhaseFoldingEngine
        self.engine = PhaseFoldingEngine()
    
    def create_test_transit_signal(
        self, 
        period: float = 10.0,
        depth: float = 0.01,
        duration: float = 0.1,
        length: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic transit signal for testing."""
        
        time = np.linspace(0, 100, length)
        flux = np.ones(length)
        
        # Add periodic transits
        phases = (time % period) / period
        transit_mask = np.abs(phases - 0.5) < (duration / 2)
        flux[transit_mask] -= depth
        
        # Add noise
        flux += np.random.normal(0, 0.001, length)
        
        return time, flux
    
    def test_fold_light_curve(self):
        """Test basic light curve folding."""
        period = 10.0
        time, flux = self.create_test_transit_signal(period=period)
        
        phase_grid, folded_flux = self.engine.fold_light_curve(
            time, flux, period, phase_bins=100
        )
        
        # Check output dimensions
        assert len(phase_grid) == 100
        assert len(folded_flux) == 100
        
        # Phase grid should span [0, 1]
        assert phase_grid[0] == 0.0
        assert phase_grid[-1] < 1.0
        
        # Transit should be visible at phase ~0.5
        center_idx = len(phase_grid) // 2
        transit_region = folded_flux[center_idx-5:center_idx+5]
        baseline_region = np.concatenate([
            folded_flux[:center_idx-10], 
            folded_flux[center_idx+10:]
        ])
        
        # Transit should be deeper than baseline
        assert np.median(transit_region) < np.median(baseline_region)
    
    def test_optimize_epoch(self):
        """Test epoch optimization."""
        period = 10.0
        time, flux = self.create_test_transit_signal(period=period)
        
        # Add random epoch offset
        epoch_offset = 2.5
        time += epoch_offset
        
        optimized_epoch, signal_strength = self.engine.optimize_epoch(
            time, flux, period
        )
        
        # Should find signal
        assert signal_strength > 0
        
        # Optimized epoch should be reasonable
        assert time[0] <= optimized_epoch <= time[-1]
    
    def test_create_dual_channel_input(self):
        """Test dual-channel input creation."""
        period = 10.0
        time, flux = self.create_test_transit_signal(period=period)
        
        raw_channel, phase_channel = self.engine.create_dual_channel_input(
            time, flux, period, target_length=512
        )
        
        # Check dimensions
        assert len(raw_channel) == 512
        assert len(phase_channel) == 512
        
        # Channels should be different (phase folding effect)
        assert not np.allclose(raw_channel, phase_channel)
    
    def test_create_dual_channel_no_period(self):
        """Test dual-channel input without period."""
        time, flux = self.create_test_transit_signal()
        
        raw_channel, phase_channel = self.engine.create_dual_channel_input(
            time, flux, period=None, target_length=512
        )
        
        # Without period, both channels should be identical
        np.testing.assert_array_equal(raw_channel, phase_channel)
    
    def test_analyze_phase_coverage(self):
        """Test phase coverage analysis."""
        period = 10.0
        time = np.linspace(0, 50, 1000)  # 5 periods
        
        coverage = self.engine.analyze_phase_coverage(time, period)
        
        # Check required keys
        required_keys = [
            'coverage_fraction', 'max_phase_gap', 'transit_coverage', 
            'n_transits', 'phase_distribution'
        ]
        for key in required_keys:
            assert key in coverage
        
        # Coverage should be reasonable
        assert 0 <= coverage['coverage_fraction'] <= 1
        assert 0 <= coverage['max_phase_gap'] <= 1
        assert coverage['n_transits'] > 0
    
    def test_calculate_signal_metrics(self):
        """Test signal quality metrics calculation."""
        # Create folded light curve with clear transit
        phase_bins = 100
        folded_flux = np.ones(phase_bins)
        
        # Add transit at center (phase 0.5)
        center = phase_bins // 2
        transit_width = 5
        folded_flux[center-transit_width:center+transit_width] -= 0.01
        
        metrics = self.engine._calculate_signal_metrics(folded_flux)
        
        # Check required keys
        required_keys = [
            'transit_depth', 'signal_to_noise', 'symmetry', 
            'baseline_flux', 'noise_level'
        ]
        for key in required_keys:
            assert key in metrics
        
        # Transit depth should be positive
        assert metrics['transit_depth'] > 0
        
        # SNR should be reasonable
        assert metrics['signal_to_noise'] > 0
    
    def test_multi_period_analysis(self):
        """Test multi-period analysis."""
        true_period = 10.0
        time, flux = self.create_test_transit_signal(period=true_period)
        
        # Test multiple periods including the true one
        test_periods = [8.0, 10.0, 12.0]
        
        results = self.engine.multi_period_analysis(time, flux, test_periods)
        
        # Should have results for all periods
        assert len(results) == len(test_periods)
        
        for period in test_periods:
            assert period in results
            
            result = results[period]
            assert 'folded_flux' in result
            assert 'signal_strength' in result
            assert 'coverage' in result
            assert 'metrics' in result
        
        # True period should have strongest signal
        true_signal = results[true_period]['signal_strength']
        for period in test_periods:
            if period != true_period:
                assert results[period]['signal_strength'] <= true_signal
    
    def test_period_detection_fallback(self):
        """Test period detection with fallback method."""
        period = 8.0
        time, flux = self.create_test_transit_signal(period=period, depth=0.02)
        
        # Test simple periodogram (fallback method)
        detected_period, power, results = self.engine._simple_periodogram(
            time, flux, period_min=5.0, period_max=15.0
        )
        
        # Should detect something close to true period
        assert 5.0 <= detected_period <= 15.0
        assert power > 0
        
        # Results should contain required keys
        assert 'periods' in results
        assert 'power' in results
        assert 'best_index' in results


class TestPreprocessingPipeline:
    """Test PreprocessingPipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from src.preprocessing.pipeline import PreprocessingPipeline
        from src.data.types import PreprocessingConfig
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = PreprocessingConfig(target_length=512)  # Smaller for testing
        
        self.pipeline = PreprocessingPipeline(
            config=self.config,
            output_dir=self.temp_dir / "processed",
            enable_phase_folding=True,
            save_intermediate=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_light_curve_file(
        self, 
        star_id: str, 
        label: int = 0,
        period: Optional[float] = None
    ) -> Path:
        """Create a test light curve file."""
        
        # Generate synthetic data
        length = 1000
        time = np.linspace(0, 100, length)
        flux = np.ones(length) + np.random.normal(0, 0.001, length)
        
        # Add transit if period provided
        if period and label == 1:
            phases = (time % period) / period
            transit_mask = np.abs(phases - 0.5) < 0.05
            flux[transit_mask] -= 0.01
        
        flux_err = np.full(length, 0.001)
        
        # Save to file
        file_path = self.temp_dir / f"{star_id}.npz"
        np.savez_compressed(
            file_path,
            time=time,
            flux=flux,
            flux_err=flux_err
        )
        
        return file_path
    
    def test_process_light_curve_file(self):
        """Test processing a single light curve file."""
        
        # Create test file
        star_id = "test_star_001"
        period = 10.0
        file_path = self.create_test_light_curve_file(star_id, label=1, period=period)
        
        # Process the file
        processed_file = self.pipeline.process_light_curve_file(
            file_path=file_path,
            star_id=star_id,
            label=1,
            period=period,
            metadata={'test': True}
        )
        
        # Check that processing succeeded
        assert processed_file is not None
        
        # Check that processed file exists
        processed_path = self.pipeline.output_dir / processed_file
        assert processed_path.exists()
        
        # Verify processed data structure
        data = np.load(processed_path)
        assert 'raw_flux' in data
        assert 'phase_folded_flux' in data
        assert 'mask' in data
        assert len(data['raw_flux']) == self.config.target_length
    
    def test_assess_quality(self):
        """Test quality assessment functionality."""
        
        # Create high-quality light curve
        high_quality_lc = LightCurve(
            star_id="high_quality",
            time=np.linspace(0, 100, 2000),  # Long
            flux=np.ones(2000) + np.random.normal(0, 0.0005, 2000),  # Low noise
            flux_err=np.full(2000, 0.0005),  # Low uncertainty
            label=1,
            period=10.0
        )
        
        # Create low-quality light curve
        low_quality_lc = LightCurve(
            star_id="low_quality", 
            time=np.linspace(0, 100, 200),  # Short
            flux=np.ones(200) + np.random.normal(0, 0.01, 200),  # High noise
            flux_err=np.full(200, 0.01),  # High uncertainty
            label=0
        )
        
        # Process both
        high_processed = self.pipeline.preprocessor.process(high_quality_lc)
        low_processed = self.pipeline.preprocessor.process(low_quality_lc)
        
        # Assess quality
        high_score = self.pipeline._assess_quality(high_quality_lc, high_processed)
        low_score = self.pipeline._assess_quality(low_quality_lc, low_processed)
        
        # High quality should score better
        assert high_score > low_score
        assert 0 <= low_score <= 1
        assert 0 <= high_score <= 1
    
    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""
        
        # Create test light curve
        lc = LightCurve(
            star_id="test_save_load",
            time=np.linspace(0, 100, 1000),
            flux=np.ones(1000) + np.random.normal(0, 0.001, 1000),
            flux_err=np.full(1000, 0.001),
            label=1,
            period=15.0
        )
        
        # Process it
        processed = self.pipeline.preprocessor.process(lc)
        
        # Save processed data
        saved_file = self.pipeline._save_processed_data(processed, lc.star_id)
        
        # Load and verify
        loaded_data = np.load(self.pipeline.output_dir / saved_file)
        
        assert 'raw_flux' in loaded_data
        assert 'phase_folded_flux' in loaded_data
        assert 'mask' in loaded_data
        assert loaded_data['label'] == processed.label
        
        # Check array shapes
        assert loaded_data['raw_flux'].shape == processed.raw_flux.shape
        assert loaded_data['phase_folded_flux'].shape == processed.phase_folded_flux.shape
        
        # Check values match
        np.testing.assert_array_almost_equal(
            loaded_data['raw_flux'], 
            processed.raw_flux.numpy()
        )
    
    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        
        summary = self.pipeline.get_pipeline_summary()
        
        # Check required sections
        assert 'configuration' in summary
        assert 'statistics' in summary
        assert 'output_directory' in summary
        
        # Check configuration details
        config = summary['configuration']
        assert config['target_length'] == self.config.target_length
        assert config['detrend_method'] == self.config.detrend_method
        assert config['enable_phase_folding'] == True
        
        # Check statistics structure
        stats = summary['statistics']
        assert 'total_processed' in stats
        assert 'successful' in stats
        assert 'failed' in stats
    
    def test_load_light_curve_from_file(self):
        """Test loading light curve from different file formats."""
        
        # Test NPZ format
        star_id = "test_load_npz"
        npz_file = self.create_test_light_curve_file(star_id)
        
        lc_npz = self.pipeline._load_light_curve_from_file(
            npz_file, star_id, label=1, period=10.0
        )
        
        assert lc_npz.star_id == star_id
        assert lc_npz.label == 1
        assert lc_npz.period == 10.0
        assert len(lc_npz.time) > 0
        assert len(lc_npz.flux) == len(lc_npz.time)
        
        # Test CSV format
        csv_file = self.temp_dir / f"{star_id}.csv"
        df = pd.DataFrame({
            'time': lc_npz.time,
            'flux': lc_npz.flux,
            'flux_err': lc_npz.flux_err
        })
        df.to_csv(csv_file, index=False)
        
        lc_csv = self.pipeline._load_light_curve_from_file(
            csv_file, star_id, label=0
        )
        
        assert lc_csv.star_id == star_id
        assert lc_csv.label == 0
        assert len(lc_csv.time) > 0
        
        # Test unsupported format
        txt_file = self.temp_dir / f"{star_id}.txt"
        with open(txt_file, 'w') as f:
            f.write("unsupported format")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.pipeline._load_light_curve_from_file(txt_file, star_id, 0)