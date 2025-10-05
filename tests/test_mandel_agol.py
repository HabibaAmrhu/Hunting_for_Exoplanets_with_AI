"""Tests for Mandel-Agol transit model."""

import pytest
import numpy as np
from src.data.types import TransitParams
from src.preprocessing.mandel_agol import MandelAgolTransitModel


class TestMandelAgolTransitModel:
    """Test MandelAgolTransitModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MandelAgolTransitModel(limb_darkening_law='quadratic')
        
        # Standard test parameters
        self.test_params = TransitParams(
            period=10.0,
            depth=1000.0,  # ppm
            duration=4.0,   # hours
            impact_parameter=0.3,
            limb_darkening=(0.4, 0.3),
            epoch=0.0
        )
    
    def test_initialization(self):
        """Test model initialization."""
        # Valid limb-darkening laws
        for law in ['linear', 'quadratic', 'nonlinear']:
            model = MandelAgolTransitModel(limb_darkening_law=law)
            assert model.limb_darkening_law == law
        
        # Invalid limb-darkening law
        with pytest.raises(ValueError, match="Unsupported limb-darkening law"):
            MandelAgolTransitModel(limb_darkening_law='invalid')
    
    def test_calculate_radius_ratio(self):
        """Test radius ratio calculation."""
        # Test with ppm depth
        params_ppm = TransitParams(
            period=10.0, depth=1000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        k_ppm = self.model._calculate_radius_ratio(params_ppm)
        expected_k = np.sqrt(1000e-6)  # sqrt(0.001)
        assert abs(k_ppm - expected_k) < 1e-6
        
        # Test with fractional depth
        params_frac = TransitParams(
            period=10.0, depth=0.001, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        k_frac = self.model._calculate_radius_ratio(params_frac)
        assert abs(k_frac - expected_k) < 1e-6
    
    def test_calculate_separation(self):
        """Test planet-star separation calculation."""
        time = np.linspace(-5, 5, 100)  # Around transit
        
        z = self.model._calculate_separation(time, self.test_params)
        
        # Check basic properties
        assert len(z) == len(time)
        assert np.all(z >= 0)  # Separation should be non-negative
        
        # At epoch (t=0), separation should be minimal
        epoch_idx = np.argmin(np.abs(time - self.test_params.epoch))
        assert z[epoch_idx] <= np.min(z) + 0.1  # Should be at or near minimum
    
    def test_uniform_disk_transit(self):
        """Test uniform disk transit calculation."""
        time = np.linspace(-1, 1, 1000)
        z = np.abs(time)  # Simple linear separation
        k = 0.1  # Small planet
        
        flux = self.model._uniform_disk(z, k)
        
        # Check basic properties
        assert len(flux) == len(z)
        assert np.all(flux <= 1.0)  # Flux should not exceed 1
        assert np.all(flux >= 1 - k**2)  # Minimum flux during complete transit
        
        # Check that deepest transit occurs at z=0
        min_flux_idx = np.argmin(flux)
        assert z[min_flux_idx] < 0.1  # Should be near z=0
    
    def test_generate_transit_basic(self):
        """Test basic transit generation."""
        time = np.linspace(-0.5, 0.5, 1000)  # Short time around transit
        
        flux = self.model.generate_transit(time, self.test_params)
        
        # Check basic properties
        assert len(flux) == len(time)
        assert np.all(flux <= 1.0)  # No flux above baseline
        assert np.min(flux) < 1.0   # Should have some transit depth
        
        # Check that transit is centered around epoch
        min_flux_idx = np.argmin(flux)
        min_time = time[min_flux_idx]
        assert abs(min_time - self.test_params.epoch) < 0.1
    
    def test_generate_transit_no_overlap(self):
        """Test transit generation when planet doesn't overlap star."""
        # Time far from transit
        time = np.linspace(5, 10, 100)  # Far from epoch=0
        
        flux = self.model.generate_transit(time, self.test_params)
        
        # Should be flat at baseline
        assert np.allclose(flux, 1.0, atol=1e-6)
    
    def test_limb_darkening_effects(self):
        """Test different limb-darkening implementations."""
        time = np.linspace(-0.2, 0.2, 500)
        
        # Compare different limb-darkening laws
        model_uniform = MandelAgolTransitModel('quadratic')
        model_linear = MandelAgolTransitModel('linear')
        
        # Parameters with no limb-darkening
        params_no_ld = TransitParams(
            period=10.0, depth=1000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.0, 0.0)
        )
        
        # Parameters with limb-darkening
        params_with_ld = TransitParams(
            period=10.0, depth=1000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        
        flux_no_ld = model_uniform.generate_transit(time, params_no_ld)
        flux_with_ld = model_uniform.generate_transit(time, params_with_ld)
        flux_linear = model_linear.generate_transit(time, params_with_ld)
        
        # Limb-darkening should affect the transit shape
        assert not np.allclose(flux_no_ld, flux_with_ld)
        assert not np.allclose(flux_with_ld, flux_linear)
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        # Valid parameters
        assert self.model.validate_parameters(self.test_params) == True
        
        # Invalid period
        invalid_period = TransitParams(
            period=-1.0, depth=1000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        assert self.model.validate_parameters(invalid_period) == False
        
        # Invalid depth
        invalid_depth = TransitParams(
            period=10.0, depth=-100.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        assert self.model.validate_parameters(invalid_depth) == False
        
        # Invalid impact parameter
        invalid_impact = TransitParams(
            period=10.0, depth=1000.0, duration=4.0,
            impact_parameter=2.0, limb_darkening=(0.4, 0.3)
        )
        assert self.model.validate_parameters(invalid_impact) == False
        
        # Invalid limb-darkening (sum > 1)
        invalid_ld = TransitParams(
            period=10.0, depth=1000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.8, 0.8)
        )
        assert self.model.validate_parameters(invalid_ld) == False
    
    def test_compute_transit_observables(self):
        """Test computation of transit observables."""
        observables = self.model.compute_transit_observables(self.test_params)
        
        # Check required keys
        required_keys = [
            'radius_ratio', 'depth_fraction', 'depth_ppm',
            'semi_major_axis_rs', 'inclination_deg', 'impact_parameter',
            'limb_darkening_u1', 'limb_darkening_u2'
        ]
        
        for key in required_keys:
            assert key in observables
        
        # Check values are reasonable
        assert 0 < observables['radius_ratio'] < 1
        assert observables['depth_ppm'] == self.test_params.depth
        assert observables['impact_parameter'] == self.test_params.impact_parameter
        assert 0 <= observables['inclination_deg'] <= 90
    
    def test_generate_transit_with_noise(self):
        """Test transit generation with noise."""
        time = np.linspace(-0.5, 0.5, 1000)
        noise_level = 0.001
        
        noisy_flux, noise = self.model.generate_transit_with_noise(
            time, self.test_params, noise_level=noise_level
        )
        
        # Check output shapes
        assert len(noisy_flux) == len(time)
        assert len(noise) == len(time)
        
        # Check noise properties
        assert np.std(noise) > 0  # Should have some noise
        assert np.std(noise) < 10 * noise_level  # But not too much
        
        # Generate clean transit for comparison
        clean_flux = self.model.generate_transit(time, self.test_params)
        
        # Noisy flux should be different from clean
        assert not np.allclose(noisy_flux, clean_flux)
        
        # But should be similar in overall shape
        correlation = np.corrcoef(noisy_flux, clean_flux)[0, 1]
        assert correlation > 0.8  # Should be well correlated
    
    def test_different_transit_depths(self):
        """Test transits with different depths."""
        time = np.linspace(-0.2, 0.2, 500)
        
        depths = [100, 1000, 5000]  # ppm
        fluxes = []
        
        for depth in depths:
            params = TransitParams(
                period=10.0, depth=depth, duration=4.0,
                impact_parameter=0.3, limb_darkening=(0.4, 0.3)
            )
            flux = self.model.generate_transit(time, params)
            fluxes.append(flux)
        
        # Deeper transits should have lower minimum flux
        min_fluxes = [np.min(flux) for flux in fluxes]
        
        # Should be in decreasing order (deeper transits have lower flux)
        for i in range(len(min_fluxes) - 1):
            assert min_fluxes[i] > min_fluxes[i + 1]
    
    def test_different_impact_parameters(self):
        """Test transits with different impact parameters."""
        time = np.linspace(-0.2, 0.2, 500)
        
        impact_params = [0.0, 0.5, 0.9]  # Central to grazing
        fluxes = []
        
        for b in impact_params:
            params = TransitParams(
                period=10.0, depth=1000.0, duration=4.0,
                impact_parameter=b, limb_darkening=(0.4, 0.3)
            )
            flux = self.model.generate_transit(time, params)
            fluxes.append(flux)
        
        # Higher impact parameter should generally result in shallower transits
        # (though this depends on limb-darkening)
        min_fluxes = [np.min(flux) for flux in fluxes]
        
        # At least check that we get different transit shapes
        for i in range(len(fluxes) - 1):
            assert not np.allclose(fluxes[i], fluxes[i + 1])
    
    def test_transit_duration_consistency(self):
        """Test that transit duration is approximately correct."""
        # Create a transit with known duration
        duration_hours = 6.0
        params = TransitParams(
            period=20.0, depth=2000.0, duration=duration_hours,
            impact_parameter=0.2, limb_darkening=(0.4, 0.3)
        )
        
        # Generate transit over longer time baseline
        time = np.linspace(-0.5, 0.5, 2000)  # Days
        flux = self.model.generate_transit(time, params)
        
        # Find points where flux is significantly below baseline
        in_transit = flux < 0.999
        
        if np.any(in_transit):
            # Estimate duration from transit points
            transit_indices = np.where(in_transit)[0]
            transit_start = time[transit_indices[0]]
            transit_end = time[transit_indices[-1]]
            estimated_duration = (transit_end - transit_start) * 24  # Convert to hours
            
            # Should be roughly consistent (within factor of 2)
            assert 0.5 * duration_hours < estimated_duration < 2.0 * duration_hours
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        time = np.linspace(-0.1, 0.1, 200)
        
        # Very small planet
        small_params = TransitParams(
            period=10.0, depth=10.0, duration=2.0,  # Very shallow
            impact_parameter=0.1, limb_darkening=(0.2, 0.1)
        )
        
        flux_small = self.model.generate_transit(time, small_params)
        assert np.all(flux_small <= 1.0)
        assert np.min(flux_small) > 0.99  # Very shallow transit
        
        # Grazing transit
        grazing_params = TransitParams(
            period=10.0, depth=1000.0, duration=8.0,
            impact_parameter=0.95, limb_darkening=(0.4, 0.3)
        )
        
        flux_grazing = self.model.generate_transit(time, grazing_params)
        assert np.all(flux_grazing <= 1.0)
        
        # Very short period (hot Jupiter)
        short_period_params = TransitParams(
            period=1.5, depth=8000.0, duration=2.0,
            impact_parameter=0.1, limb_darkening=(0.3, 0.2)
        )
        
        flux_short = self.model.generate_transit(time, short_period_params)
        assert np.all(flux_short <= 1.0)
        assert np.min(flux_short) < 0.995  # Should see significant transit


class TestParameterSampling:
    """Test parameter sampling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.preprocessing.parameter_sampling import (
            StellarParameterSampler, 
            ExoplanetParameterSampler,
            TransitParameterGenerator
        )
        
        self.stellar_sampler = StellarParameterSampler('kepler')
        self.planet_sampler = ExoplanetParameterSampler('kepler')
        self.generator = TransitParameterGenerator('kepler', 'kepler')
    
    def test_stellar_parameter_sampling(self):
        """Test stellar parameter sampling."""
        n_samples = 100
        stellar_params = self.stellar_sampler.sample_stellar_parameters(n_samples)
        
        # Check required keys
        required_keys = ['teff', 'logg', 'feh', 'radius', 'mass']
        for key in required_keys:
            assert key in stellar_params
            assert len(stellar_params[key]) == n_samples
        
        # Check parameter ranges are reasonable
        assert np.all(stellar_params['teff'] >= 3000)
        assert np.all(stellar_params['teff'] <= 8000)
        assert np.all(stellar_params['logg'] >= 3.0)
        assert np.all(stellar_params['logg'] <= 5.5)
        assert np.all(stellar_params['radius'] > 0)
        assert np.all(stellar_params['mass'] > 0)
    
    def test_limb_darkening_coefficients(self):
        """Test limb-darkening coefficient calculation."""
        # Test single values
        teff = 5778  # Solar temperature
        u1, u2 = self.stellar_sampler.get_limb_darkening_coefficients(teff)
        
        assert 0 <= u1 <= 1
        assert 0 <= u2 <= 1
        assert u1 + u2 <= 1
        
        # Test arrays
        teff_array = np.array([4000, 5778, 7000])
        u1_array, u2_array = self.stellar_sampler.get_limb_darkening_coefficients(teff_array)
        
        assert len(u1_array) == 3
        assert len(u2_array) == 3
        assert np.all(u1_array >= 0) and np.all(u1_array <= 1)
        assert np.all(u2_array >= 0) and np.all(u2_array <= 1)
        assert np.all(u1_array + u2_array <= 1)
        
        # Cooler stars should have different coefficients than hotter stars
        assert u1_array[0] != u1_array[2]  # 4000K vs 7000K should be different
    
    def test_planet_parameter_sampling(self):
        """Test planet parameter sampling."""
        # Create mock stellar parameters
        stellar_params = {
            'teff': np.array([5778]),
            'logg': np.array([4.4]),
            'feh': np.array([0.0]),
            'radius': np.array([1.0]),
            'mass': np.array([1.0])
        }
        
        n_samples = 50
        planet_params = self.planet_sampler.sample_planet_parameters(stellar_params, n_samples)
        
        # Check required keys
        required_keys = ['periods', 'planet_radii_earth', 'impact_parameters', 
                        'durations_hours', 'depths_ppm']
        for key in required_keys:
            assert key in planet_params
            assert len(planet_params[key]) == n_samples
        
        # Check parameter ranges
        assert np.all(planet_params['periods'] > 0)
        assert np.all(planet_params['planet_radii_earth'] > 0)
        assert np.all(planet_params['impact_parameters'] >= 0)
        assert np.all(planet_params['durations_hours'] > 0)
        assert np.all(planet_params['depths_ppm'] > 0)
        
        # Check reasonable ranges
        assert np.all(planet_params['periods'] < 1000)  # Less than ~3 years
        assert np.all(planet_params['durations_hours'] < 48)  # Less than 2 days
    
    def test_transit_parameter_generation(self):
        """Test complete transit parameter generation."""
        n_samples = 20
        params_list = self.generator.generate_transit_parameters(n_samples, seed=42)
        
        assert len(params_list) == n_samples
        
        for params in params_list:
            # Check type
            assert isinstance(params, TransitParams)
            
            # Check parameter validity
            assert params.period > 0
            assert params.depth > 0
            assert params.duration > 0
            assert 0 <= params.impact_parameter <= 1.5
            assert 0 <= params.limb_darkening[0] <= 1
            assert 0 <= params.limb_darkening[1] <= 1
            assert params.limb_darkening[0] + params.limb_darkening[1] <= 1
            assert 0 <= params.epoch < params.period
    
    def test_parameter_statistics(self):
        """Test parameter statistics generation."""
        stats = self.generator.generate_parameter_statistics(n_samples=1000)
        
        # Check required sections
        required_sections = [
            'periods', 'depths_ppm', 'durations_hours', 
            'impact_parameters', 'limb_darkening_u1', 'limb_darkening_u2'
        ]
        
        for section in required_sections:
            assert section in stats
            
            # Check required statistics
            required_stats = ['mean', 'std', 'median', 'min', 'max']
            for stat in required_stats:
                assert stat in stats[section]
                assert isinstance(stats[section][stat], float)
        
        # Check that statistics are reasonable
        assert stats['periods']['mean'] > 0
        assert stats['depths_ppm']['mean'] > 0
        assert stats['durations_hours']['mean'] > 0
        assert 0 <= stats['impact_parameters']['mean'] <= 1
    
    def test_different_catalog_types(self):
        """Test different stellar catalog types."""
        catalogs = ['kepler', 'tess', 'generic']
        
        for catalog in catalogs:
            sampler = StellarParameterSampler(catalog)
            params = sampler.sample_stellar_parameters(10)
            
            # Should produce valid parameters for all catalog types
            assert len(params['teff']) == 10
            assert np.all(params['teff'] > 0)
            assert np.all(params['radius'] > 0)
            assert np.all(params['mass'] > 0)
    
    def test_different_survey_types(self):
        """Test different planet survey types."""
        surveys = ['kepler', 'tess', 'combined']
        
        # Mock stellar parameters
        stellar_params = {
            'teff': np.array([5778]),
            'logg': np.array([4.4]),
            'feh': np.array([0.0]),
            'radius': np.array([1.0]),
            'mass': np.array([1.0])
        }
        
        for survey in surveys:
            sampler = ExoplanetParameterSampler(survey)
            params = sampler.sample_planet_parameters(stellar_params, 10)
            
            # Should produce valid parameters for all survey types
            assert len(params['periods']) == 10
            assert np.all(params['periods'] > 0)
            assert np.all(params['depths_ppm'] > 0)
    
    def test_reproducibility(self):
        """Test that sampling is reproducible with seeds."""
        seed = 12345
        
        # Generate parameters twice with same seed
        params1 = self.generator.generate_transit_parameters(5, seed=seed)
        params2 = self.generator.generate_transit_parameters(5, seed=seed)
        
        # Should be identical
        for p1, p2 in zip(params1, params2):
            assert p1.period == p2.period
            assert p1.depth == p2.depth
            assert p1.duration == p2.duration
            assert p1.impact_parameter == p2.impact_parameter
            assert p1.limb_darkening == p2.limb_darkening
            assert p1.epoch == p2.epoch
    
    def test_parameter_correlations(self):
        """Test that parameter correlations are reasonable."""
        n_samples = 200
        params_list = self.generator.generate_transit_parameters(n_samples, seed=42)
        
        # Extract arrays
        periods = np.array([p.period for p in params_list])
        durations = np.array([p.duration for p in params_list])
        depths = np.array([p.depth for p in params_list])
        
        # Longer periods should generally have longer durations
        correlation_p_d = np.corrcoef(periods, durations)[0, 1]
        assert correlation_p_d > 0.1  # Should be some positive correlation
        
        # Check that we have a reasonable range of values
        assert np.std(periods) > 0  # Should have variety
        assert np.std(depths) > 0
        assert np.std(durations) > 0


class TestSyntheticTransitInjector:
    """Test SyntheticTransitInjector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.preprocessing.synthetic_injection import SyntheticTransitInjector
        
        self.injector = SyntheticTransitInjector(
            stellar_catalog='kepler',
            planet_survey='kepler',
            noise_model='gaussian'
        )
    
    def create_test_light_curve(
        self, 
        star_id: str = "test_star",
        length: int = 1000,
        has_planet: bool = False
    ) -> LightCurve:
        """Create a test light curve."""
        
        time = np.linspace(0, 100, length)
        flux = np.ones(length) + np.random.normal(0, 0.001, length)
        flux_err = np.full(length, 0.001)
        
        metadata = {
            'teff': 5778,
            'radius': 1.0,
            'mass': 1.0,
            'logg': 4.4,
            'feh': 0.0
        }
        
        return LightCurve(
            star_id=star_id,
            time=time,
            flux=flux,
            flux_err=flux_err,
            label=1 if has_planet else 0,
            metadata=metadata
        )
    
    def test_inject_transit_basic(self):
        """Test basic transit injection."""
        # Create non-planet light curve
        lc = self.create_test_light_curve("test_star", has_planet=False)
        
        # Inject transit
        injected_lc, metadata = self.injector.inject_transit(lc)
        
        # Check that injection was successful
        assert metadata['success'] == True
        assert injected_lc.label == 1  # Should now be a planet
        assert injected_lc.star_id != lc.star_id  # Should have modified ID
        assert 'synthetic_injection' in injected_lc.metadata
        assert injected_lc.metadata['synthetic_injection'] == True
        
        # Check that flux was modified
        assert not np.allclose(injected_lc.flux, lc.flux)
        
        # Check that some flux values are lower (transit effect)
        assert np.min(injected_lc.flux) < np.min(lc.flux)
    
    def test_inject_transit_with_custom_parameters(self):
        """Test transit injection with custom parameters."""
        lc = self.create_test_light_curve("test_star", has_planet=False)
        
        # Create custom transit parameters
        custom_params = TransitParams(
            period=15.0,
            depth=2000.0,  # ppm
            duration=6.0,   # hours
            impact_parameter=0.2,
            limb_darkening=(0.4, 0.3),
            epoch=5.0
        )
        
        # Inject with custom parameters
        injected_lc, metadata = self.injector.inject_transit(lc, custom_params)
        
        # Check success
        assert metadata['success'] == True
        assert injected_lc.period == custom_params.period
        
        # Check that parameters were recorded
        recorded_params = metadata['transit_params']
        assert recorded_params['period'] == custom_params.period
        assert recorded_params['depth'] == custom_params.depth
    
    def test_extract_stellar_info(self):
        """Test stellar information extraction."""
        # Light curve with complete metadata
        lc_complete = self.create_test_light_curve("complete")
        stellar_info = self.injector._extract_stellar_info(lc_complete)
        
        assert stellar_info['teff'] == 5778
        assert stellar_info['radius'] == 1.0
        assert stellar_info['mass'] == 1.0
        
        # Light curve with minimal metadata
        lc_minimal = LightCurve(
            star_id="minimal",
            time=np.linspace(0, 100, 1000),
            flux=np.ones(1000) + np.random.normal(0, 0.005, 1000),  # Higher variability
            flux_err=np.full(1000, 0.001),
            label=0,
            metadata={}  # Empty metadata
        )
        
        stellar_info_minimal = self.injector._extract_stellar_info(lc_minimal)
        
        # Should have estimated values
        assert stellar_info_minimal['teff'] is not None
        assert stellar_info_minimal['radius'] == 1.0  # Default
        assert stellar_info_minimal['mass'] == 1.0    # Default
    
    def test_assess_injection_quality(self):
        """Test injection quality assessment."""
        original_lc = self.create_test_light_curve("original", has_planet=False)
        
        # Create injected light curve (simulate injection)
        injected_flux = original_lc.flux.copy()
        injected_flux[400:450] -= 0.01  # Simulate transit
        
        injected_lc = LightCurve(
            star_id="injected",
            time=original_lc.time,
            flux=injected_flux,
            flux_err=original_lc.flux_err,
            label=1,
            metadata=original_lc.metadata
        )
        
        # Create mock transit parameters
        transit_params = TransitParams(
            period=20.0, depth=10000.0, duration=4.0,
            impact_parameter=0.3, limb_darkening=(0.4, 0.3)
        )
        
        # Create mock synthetic transit
        synthetic_transit = np.ones_like(original_lc.flux)
        synthetic_transit[400:450] = 0.99  # 1% depth
        
        # Assess quality
        quality = self.injector._assess_injection_quality(
            original_lc, injected_lc, transit_params, synthetic_transit
        )
        
        # Check required metrics
        required_metrics = [
            'snr', 'detectability', 'depth_fidelity', 'overall_quality',
            'transit_depth_achieved', 'transit_depth_expected'
        ]
        
        for metric in required_metrics:
            assert metric in quality
            assert isinstance(quality[metric], float)
        
        # Check reasonable values
        assert quality['snr'] > 0
        assert 0 <= quality['overall_quality'] <= 1
    
    def test_balance_dataset(self):
        """Test dataset balancing functionality."""
        # Create unbalanced dataset
        light_curves = []
        
        # Add many non-planet light curves
        for i in range(20):
            lc = self.create_test_light_curve(f"non_planet_{i}", has_planet=False)
            light_curves.append(lc)
        
        # Add few planet light curves
        for i in range(3):
            lc = self.create_test_light_curve(f"planet_{i}", has_planet=True)
            light_curves.append(lc)
        
        # Balance dataset
        balanced_lcs, stats = self.injector.balance_dataset(
            light_curves, target_ratio=0.4, max_injections_per_star=2
        )
        
        # Check balancing statistics
        assert 'injections_made' in stats
        assert 'final_ratio' in stats
        assert 'injection_success_rate' in stats
        
        # Check that we have more planets now
        final_planets = len([lc for lc in balanced_lcs if lc.label == 1])
        original_planets = len([lc for lc in light_curves if lc.label == 1])
        
        assert final_planets > original_planets
        
        # Check that ratio is closer to target
        final_ratio = final_planets / len(balanced_lcs)
        assert abs(final_ratio - 0.4) < abs(original_planets / len(light_curves) - 0.4)
    
    def test_different_noise_models(self):
        """Test different noise models."""
        lc = self.create_test_light_curve("test_noise", has_planet=False)
        
        noise_models = ['none', 'gaussian', 'realistic']
        
        for noise_model in noise_models:
            injector = SyntheticTransitInjector(noise_model=noise_model)
            injected_lc, metadata = injector.inject_transit(lc)
            
            # Should succeed for all noise models
            assert metadata['success'] == True
            assert injected_lc.label == 1
    
    def test_injection_with_existing_planet(self):
        """Test injection into light curve that already has a planet."""
        # Create light curve with existing planet
        lc = self.create_test_light_curve("existing_planet", has_planet=True)
        
        # Should still work but give warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            injected_lc, metadata = self.injector.inject_transit(lc)
            
            # Check that warning was issued
            assert len(w) > 0
            assert "already has label=1" in str(w[0].message)
        
        # Should still succeed
        assert metadata['success'] == True
    
    def test_create_injection_report(self):
        """Test injection report creation."""
        # Perform some injections first
        lc = self.create_test_light_curve("report_test", has_planet=False)
        
        for i in range(5):
            self.injector.inject_transit(lc)
        
        # Create report
        report = self.injector.create_injection_report()
        
        # Check report structure
        assert 'injection_summary' in report
        assert 'configuration' in report
        assert 'quality_analysis' in report
        
        # Check injection summary
        summary = report['injection_summary']
        assert summary['total_injections'] == 5
        assert summary['successful_injections'] > 0
        
        # Check configuration
        config = report['configuration']
        assert config['stellar_catalog'] == 'kepler'
        assert config['planet_survey'] == 'kepler'
        assert config['noise_model'] == 'gaussian'
    
    def test_validate_injected_transits(self):
        """Test validation of injected transits."""
        # Create dataset with injected transits
        light_curves = []
        
        for i in range(10):
            lc = self.create_test_light_curve(f"validation_{i}", has_planet=False)
            injected_lc, _ = self.injector.inject_transit(lc)
            light_curves.append(injected_lc)
        
        # Validate injections
        validation_report = self.injector.validate_injected_transits(
            light_curves, validation_fraction=0.5
        )
        
        # Check validation report
        assert 'n_validated' in validation_report
        assert 'detection_rate' in validation_report
        assert 'n_total_injected' in validation_report
        
        # Should have validated some light curves
        assert validation_report['n_validated'] > 0
        assert validation_report['n_total_injected'] == 10
    
    def test_injection_failure_handling(self):
        """Test handling of injection failures."""
        # Create problematic light curve (very short)
        problematic_lc = LightCurve(
            star_id="problematic",
            time=np.array([0, 1]),  # Only 2 points
            flux=np.array([1.0, 1.0]),
            flux_err=np.array([0.001, 0.001]),
            label=0,
            metadata={}
        )
        
        # Try to inject (should handle gracefully)
        result_lc, metadata = self.injector.inject_transit(problematic_lc)
        
        # Should return original light curve and failure metadata
        assert metadata['success'] == False
        assert 'error' in metadata
        assert result_lc.star_id == problematic_lc.star_id  # Should be unchanged