"""
Synthetic transit injection pipeline for physics-informed data augmentation.

Combines the Mandel-Agol transit model with realistic parameter sampling
to inject synthetic transits into real light curves for class balancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import json

from ..data.types import LightCurve, TransitParams
from .mandel_agol import MandelAgolTransitModel
from .parameter_sampling import TransitParameterGenerator


class SyntheticTransitInjector:
    """
    Injects physics-based synthetic transits into real light curves.
    
    Uses the Mandel-Agol model with realistic parameter distributions
    to create balanced datasets for exoplanet detection training.
    """
    
    def __init__(
        self,
        stellar_catalog: str = 'kepler',
        planet_survey: str = 'kepler',
        limb_darkening_law: str = 'quadratic',
        noise_model: str = 'realistic'
    ):
        """
        Initialize synthetic transit injector.
        
        Args:
            stellar_catalog: Stellar parameter distribution type
            planet_survey: Planet parameter distribution type  
            limb_darkening_law: Limb-darkening model to use
            noise_model: Noise model ('realistic', 'gaussian', 'none')
        """
        self.stellar_catalog = stellar_catalog
        self.planet_survey = planet_survey
        self.noise_model = noise_model
        
        # Initialize components
        self.transit_model = MandelAgolTransitModel(limb_darkening_law)
        self.parameter_generator = TransitParameterGenerator(stellar_catalog, planet_survey)
        
        # Statistics tracking
        self.injection_stats = {
            'total_injections': 0,
            'successful_injections': 0,
            'failed_injections': 0,
            'parameter_stats': {},
            'quality_scores': []
        }
    
    def inject_transit(
        self,
        light_curve: LightCurve,
        transit_params: Optional[TransitParams] = None,
        preserve_original: bool = True
    ) -> Tuple[LightCurve, Dict]:
        """
        Inject a synthetic transit into a light curve.
        
        Args:
            light_curve: Original light curve (should be non-planet)
            transit_params: Transit parameters (if None, will be sampled)
            preserve_original: Whether to preserve original light curve properties
            
        Returns:
            Tuple of (modified_light_curve, injection_metadata)
        """
        if light_curve.label == 1:
            warnings.warn("Injecting transit into light curve that already has label=1")
        
        try:
            # Generate transit parameters if not provided
            if transit_params is None:
                # Use stellar properties from light curve metadata if available
                stellar_info = self._extract_stellar_info(light_curve)
                transit_params = self._generate_compatible_parameters(stellar_info)
            
            # Validate parameters
            if not self.transit_model.validate_parameters(transit_params):
                raise ValueError("Invalid transit parameters")
            
            # Generate synthetic transit
            synthetic_transit = self.transit_model.generate_transit(
                light_curve.time, transit_params
            )
            
            # Add noise if requested
            if self.noise_model != 'none':
                synthetic_transit = self._add_realistic_noise(
                    synthetic_transit, light_curve, transit_params
                )
            
            # Inject transit into original flux
            injected_flux = light_curve.flux * synthetic_transit
            
            # Create new light curve
            injected_lc = LightCurve(
                star_id=f"{light_curve.star_id}_injected",
                time=light_curve.time.copy(),
                flux=injected_flux,
                flux_err=light_curve.flux_err.copy(),
                label=1,  # Now has a planet
                period=transit_params.period,
                metadata={
                    **light_curve.metadata,
                    'synthetic_injection': True,
                    'original_star_id': light_curve.star_id,
                    'injection_params': transit_params.__dict__
                }
            )
            
            # Calculate injection quality metrics
            quality_metrics = self._assess_injection_quality(
                light_curve, injected_lc, transit_params, synthetic_transit
            )
            
            # Create injection metadata
            injection_metadata = {
                'success': True,
                'transit_params': transit_params.__dict__,
                'quality_metrics': quality_metrics,
                'original_label': light_curve.label,
                'injected_label': injected_lc.label,
                'stellar_info': stellar_info
            }
            
            self.injection_stats['successful_injections'] += 1
            self.injection_stats['quality_scores'].append(quality_metrics['overall_quality'])
            
            return injected_lc, injection_metadata
            
        except Exception as e:
            self.injection_stats['failed_injections'] += 1
            
            injection_metadata = {
                'success': False,
                'error': str(e),
                'original_label': light_curve.label
            }
            
            return light_curve, injection_metadata
        
        finally:
            self.injection_stats['total_injections'] += 1
    
    def _extract_stellar_info(self, light_curve: LightCurve) -> Dict:
        """Extract stellar information from light curve metadata."""
        
        stellar_info = {}
        
        # Try to extract from metadata
        if light_curve.metadata:
            stellar_info['teff'] = light_curve.metadata.get('teff', None)
            stellar_info['radius'] = light_curve.metadata.get('radius', None)
            stellar_info['mass'] = light_curve.metadata.get('mass', None)
            stellar_info['logg'] = light_curve.metadata.get('logg', None)
            stellar_info['feh'] = light_curve.metadata.get('feh', None)
        
        # Estimate from light curve properties if not available
        if stellar_info.get('teff') is None:
            # Rough estimate from variability (very approximate)
            flux_std = np.std(light_curve.flux)
            if flux_std < 0.001:
                stellar_info['teff'] = 6000  # Quiet, sun-like
            elif flux_std < 0.005:
                stellar_info['teff'] = 5500  # Moderate activity
            else:
                stellar_info['teff'] = 4500  # Active, cooler star
        
        # Set defaults for missing values
        stellar_info.setdefault('radius', 1.0)  # Solar radii
        stellar_info.setdefault('mass', 1.0)    # Solar masses
        stellar_info.setdefault('logg', 4.4)    # Solar log g
        stellar_info.setdefault('feh', 0.0)     # Solar metallicity
        
        return stellar_info
    
    def _generate_compatible_parameters(self, stellar_info: Dict) -> TransitParams:
        """Generate transit parameters compatible with stellar properties."""
        
        # Create temporary stellar parameter arrays for the generator
        stellar_params = {
            'teff': np.array([stellar_info['teff']]),
            'logg': np.array([stellar_info['logg']]),
            'feh': np.array([stellar_info['feh']]),
            'radius': np.array([stellar_info['radius']]),
            'mass': np.array([stellar_info['mass']])
        }
        
        # Generate parameters using the stellar properties
        params_list = self.parameter_generator.generate_transit_parameters(1)
        
        # Adjust the generated parameters based on stellar properties
        params = params_list[0]
        
        # Recalculate limb-darkening for this specific star
        u1, u2 = self.parameter_generator.stellar_sampler.get_limb_darkening_coefficients(
            stellar_info['teff'], stellar_info['logg'], stellar_info['feh']
        )
        
        # Update limb-darkening coefficients
        params.limb_darkening = (float(u1[0]), float(u2[0]))
        
        return params
    
    def _add_realistic_noise(
        self,
        synthetic_transit: np.ndarray,
        original_lc: LightCurve,
        transit_params: TransitParams
    ) -> np.ndarray:
        """Add realistic noise to synthetic transit."""
        
        if self.noise_model == 'gaussian':
            # Simple Gaussian noise based on original light curve
            noise_level = np.std(original_lc.flux) * 0.1  # 10% of original variability
            noise = np.random.normal(0, noise_level, len(synthetic_transit))
            return synthetic_transit + noise
            
        elif self.noise_model == 'realistic':
            # More sophisticated noise model
            
            # Base noise level from original light curve uncertainties
            if hasattr(original_lc, 'flux_err') and original_lc.flux_err is not None:
                base_noise_level = np.median(original_lc.flux_err)
            else:
                base_noise_level = np.std(original_lc.flux) * 0.05
            
            # Gaussian photon noise
            photon_noise = np.random.normal(0, base_noise_level, len(synthetic_transit))
            
            # Correlated noise (systematic effects)
            # Add slow variations
            time_norm = (original_lc.time - original_lc.time[0]) / (original_lc.time[-1] - original_lc.time[0])
            systematic_noise = 0.0002 * np.sin(2 * np.pi * time_norm * 3)  # 3 cycles over observation
            
            # Add faster variations
            systematic_noise += 0.0001 * np.sin(2 * np.pi * time_norm * 20)  # 20 cycles
            
            # Combine noise sources
            total_noise = photon_noise + systematic_noise
            
            return synthetic_transit + total_noise
        
        else:  # 'none'
            return synthetic_transit
    
    def _assess_injection_quality(
        self,
        original_lc: LightCurve,
        injected_lc: LightCurve,
        transit_params: TransitParams,
        synthetic_transit: np.ndarray
    ) -> Dict:
        """Assess the quality of transit injection."""
        
        # Signal-to-noise ratio of injected transit
        baseline_flux = np.median(synthetic_transit)
        transit_depth = baseline_flux - np.min(synthetic_transit)
        
        if hasattr(original_lc, 'flux_err') and original_lc.flux_err is not None:
            noise_level = np.median(original_lc.flux_err)
        else:
            noise_level = np.std(original_lc.flux)
        
        snr = transit_depth / noise_level if noise_level > 0 else 0
        
        # Transit detectability (rough estimate)
        # Based on transit depth relative to stellar variability
        stellar_variability = np.std(original_lc.flux)
        detectability = transit_depth / stellar_variability if stellar_variability > 0 else 0
        
        # Injection fidelity (how well the transit was preserved)
        expected_depth = transit_params.depth * 1e-6 if transit_params.depth > 1 else transit_params.depth
        depth_fidelity = min(transit_depth / expected_depth, 1.0) if expected_depth > 0 else 0
        
        # Overall quality score
        quality_factors = [
            min(snr / 10.0, 1.0),        # SNR factor (normalized to 10)
            min(detectability / 3.0, 1.0), # Detectability factor
            depth_fidelity,               # Depth preservation
            1.0 if transit_params.duration > 1.0 else 0.5  # Duration reasonableness
        ]
        
        overall_quality = np.mean(quality_factors)
        
        return {
            'snr': float(snr),
            'detectability': float(detectability),
            'depth_fidelity': float(depth_fidelity),
            'overall_quality': float(overall_quality),
            'transit_depth_achieved': float(transit_depth),
            'transit_depth_expected': float(expected_depth),
            'stellar_variability': float(stellar_variability),
            'noise_level': float(noise_level)
        }
    
    def balance_dataset(
        self,
        light_curves: List[LightCurve],
        target_ratio: float = 0.5,
        max_injections_per_star: int = 3
    ) -> Tuple[List[LightCurve], Dict]:
        """
        Balance dataset by injecting synthetic transits.
        
        Args:
            light_curves: List of light curves to balance
            target_ratio: Target fraction of planet light curves
            max_injections_per_star: Maximum injections per original light curve
            
        Returns:
            Tuple of (balanced_light_curves, balancing_statistics)
        """
        # Separate planet and non-planet light curves
        planet_lcs = [lc for lc in light_curves if lc.label == 1]
        non_planet_lcs = [lc for lc in light_curves if lc.label == 0]
        
        n_planets = len(planet_lcs)
        n_non_planets = len(non_planet_lcs)
        total_original = n_planets + n_non_planets
        
        print(f"Original dataset: {n_planets} planets, {n_non_planets} non-planets")
        
        # Calculate how many synthetic planets we need
        target_planets = int(total_original * target_ratio / (1 - target_ratio))
        needed_planets = max(0, target_planets - n_planets)
        
        print(f"Target: {target_planets} total planets, need to inject: {needed_planets}")
        
        if needed_planets == 0:
            return light_curves, {'injections_needed': 0, 'injections_made': 0}
        
        # Reset injection statistics
        self.injection_stats = {
            'total_injections': 0,
            'successful_injections': 0,
            'failed_injections': 0,
            'parameter_stats': {},
            'quality_scores': []
        }
        
        # Inject synthetic transits
        balanced_lcs = light_curves.copy()
        injections_made = 0
        
        # Randomly select non-planet light curves for injection
        np.random.shuffle(non_planet_lcs)
        
        for i, original_lc in enumerate(non_planet_lcs):
            if injections_made >= needed_planets:
                break
            
            # Limit injections per star
            injections_this_star = min(
                max_injections_per_star,
                needed_planets - injections_made
            )
            
            for j in range(injections_this_star):
                try:
                    injected_lc, metadata = self.inject_transit(original_lc)
                    
                    if metadata['success']:
                        balanced_lcs.append(injected_lc)
                        injections_made += 1
                        
                        if injections_made >= needed_planets:
                            break
                            
                except Exception as e:
                    warnings.warn(f"Failed to inject transit into {original_lc.star_id}: {e}")
                    continue
        
        # Calculate final statistics
        final_planets = len([lc for lc in balanced_lcs if lc.label == 1])
        final_non_planets = len([lc for lc in balanced_lcs if lc.label == 0])
        final_ratio = final_planets / len(balanced_lcs)
        
        balancing_stats = {
            'original_planets': n_planets,
            'original_non_planets': n_non_planets,
            'injections_needed': needed_planets,
            'injections_made': injections_made,
            'final_planets': final_planets,
            'final_non_planets': final_non_planets,
            'final_ratio': final_ratio,
            'target_ratio': target_ratio,
            'injection_success_rate': (
                self.injection_stats['successful_injections'] / 
                max(self.injection_stats['total_injections'], 1)
            ),
            'average_quality': np.mean(self.injection_stats['quality_scores']) if self.injection_stats['quality_scores'] else 0
        }
        
        print(f"Balancing complete: {final_planets} planets, {final_non_planets} non-planets")
        print(f"Final ratio: {final_ratio:.3f}, Success rate: {balancing_stats['injection_success_rate']:.3f}")
        
        return balanced_lcs, balancing_stats
    
    def create_injection_report(
        self,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Create detailed report of injection statistics.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Dictionary with injection report
        """
        report = {
            'injection_summary': self.injection_stats.copy(),
            'configuration': {
                'stellar_catalog': self.stellar_catalog,
                'planet_survey': self.planet_survey,
                'noise_model': self.noise_model,
                'limb_darkening_law': self.transit_model.limb_darkening_law
            },
            'quality_analysis': {}
        }
        
        # Analyze quality scores
        if self.injection_stats['quality_scores']:
            quality_scores = np.array(self.injection_stats['quality_scores'])
            
            report['quality_analysis'] = {
                'mean_quality': float(np.mean(quality_scores)),
                'std_quality': float(np.std(quality_scores)),
                'min_quality': float(np.min(quality_scores)),
                'max_quality': float(np.max(quality_scores)),
                'median_quality': float(np.median(quality_scores)),
                'high_quality_fraction': float(np.sum(quality_scores > 0.7) / len(quality_scores)),
                'low_quality_fraction': float(np.sum(quality_scores < 0.3) / len(quality_scores))
            }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Injection report saved to: {output_path}")
        
        return report
    
    def validate_injected_transits(
        self,
        injected_light_curves: List[LightCurve],
        validation_fraction: float = 0.1
    ) -> Dict:
        """
        Validate quality of injected transits by analyzing a sample.
        
        Args:
            injected_light_curves: List of light curves with injected transits
            validation_fraction: Fraction of light curves to validate
            
        Returns:
            Validation report dictionary
        """
        # Select random sample for validation
        injected_lcs = [lc for lc in injected_light_curves 
                       if lc.metadata.get('synthetic_injection', False)]
        
        n_validate = max(1, int(len(injected_lcs) * validation_fraction))
        validation_sample = np.random.choice(injected_lcs, n_validate, replace=False)
        
        validation_results = {
            'n_validated': n_validate,
            'n_total_injected': len(injected_lcs),
            'transit_detected': 0,
            'transit_depth_errors': [],
            'period_consistency': [],
            'snr_estimates': []
        }
        
        for lc in validation_sample:
            try:
                # Extract injection parameters
                injection_params = lc.metadata.get('injection_params', {})
                expected_period = injection_params.get('period')
                expected_depth = injection_params.get('depth')
                
                if expected_period and expected_depth:
                    # Simple transit detection test
                    # Fold light curve at expected period
                    phases = (lc.time % expected_period) / expected_period
                    
                    # Check for transit signal at phase 0.5
                    transit_phases = phases[(phases > 0.4) & (phases < 0.6)]
                    baseline_phases = phases[(phases < 0.3) | (phases > 0.7)]
                    
                    if len(transit_phases) > 0 and len(baseline_phases) > 0:
                        transit_flux = lc.flux[(phases > 0.4) & (phases < 0.6)]
                        baseline_flux = lc.flux[(phases < 0.3) | (phases > 0.7)]
                        
                        transit_median = np.median(transit_flux)
                        baseline_median = np.median(baseline_flux)
                        
                        # Check if transit is detected
                        if baseline_median - transit_median > 0:
                            validation_results['transit_detected'] += 1
                            
                            # Calculate depth error
                            observed_depth = (baseline_median - transit_median) * 1e6  # ppm
                            depth_error = abs(observed_depth - expected_depth) / expected_depth
                            validation_results['transit_depth_errors'].append(depth_error)
                            
                            # Estimate SNR
                            noise_level = np.std(baseline_flux)
                            snr = (baseline_median - transit_median) / noise_level
                            validation_results['snr_estimates'].append(snr)
                
            except Exception as e:
                warnings.warn(f"Validation failed for {lc.star_id}: {e}")
                continue
        
        # Calculate summary statistics
        validation_results['detection_rate'] = (
            validation_results['transit_detected'] / n_validate
        )
        
        if validation_results['transit_depth_errors']:
            validation_results['mean_depth_error'] = np.mean(validation_results['transit_depth_errors'])
            validation_results['std_depth_error'] = np.std(validation_results['transit_depth_errors'])
        
        if validation_results['snr_estimates']:
            validation_results['mean_snr'] = np.mean(validation_results['snr_estimates'])
            validation_results['median_snr'] = np.median(validation_results['snr_estimates'])
        
        return validation_results