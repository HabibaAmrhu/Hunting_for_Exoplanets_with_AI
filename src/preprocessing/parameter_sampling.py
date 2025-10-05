"""
Realistic parameter sampling for physics-informed synthetic transit generation.

Based on observed exoplanet populations from Kepler, TESS, and ground-based surveys.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

from ..data.types import TransitParams


class StellarParameterSampler:
    """
    Samples realistic stellar parameters based on observed stellar populations.
    
    Uses empirical distributions from stellar catalogs to generate
    realistic host star properties for synthetic transit generation.
    """
    
    def __init__(self, catalog_type: str = 'kepler'):
        """
        Initialize stellar parameter sampler.
        
        Args:
            catalog_type: Type of stellar catalog ('kepler', 'tess', 'generic')
        """
        self.catalog_type = catalog_type
        self._initialize_distributions()
    
    def _initialize_distributions(self):
        """Initialize empirical stellar parameter distributions."""
        
        if self.catalog_type == 'kepler':
            # Kepler stellar sample characteristics
            self.teff_params = {'loc': 5778, 'scale': 800, 'min': 3500, 'max': 7500}
            self.logg_params = {'loc': 4.4, 'scale': 0.3, 'min': 3.5, 'max': 5.0}
            self.feh_params = {'loc': 0.0, 'scale': 0.2, 'min': -1.0, 'max': 0.5}
            self.radius_params = {'loc': 1.0, 'scale': 0.3, 'min': 0.5, 'max': 2.5}
            self.mass_params = {'loc': 1.0, 'scale': 0.2, 'min': 0.6, 'max': 1.8}
            
        elif self.catalog_type == 'tess':
            # TESS all-sky sample (broader range)
            self.teff_params = {'loc': 5200, 'scale': 1200, 'min': 3000, 'max': 8000}
            self.logg_params = {'loc': 4.3, 'scale': 0.4, 'min': 3.0, 'max': 5.2}
            self.feh_params = {'loc': -0.1, 'scale': 0.3, 'min': -1.5, 'max': 0.8}
            self.radius_params = {'loc': 1.1, 'scale': 0.5, 'min': 0.3, 'max': 4.0}
            self.mass_params = {'loc': 0.9, 'scale': 0.3, 'min': 0.3, 'max': 2.5}
            
        else:  # generic
            # Generic stellar population
            self.teff_params = {'loc': 5500, 'scale': 1000, 'min': 3200, 'max': 7800}
            self.logg_params = {'loc': 4.4, 'scale': 0.4, 'min': 3.2, 'max': 5.1}
            self.feh_params = {'loc': 0.0, 'scale': 0.25, 'min': -1.2, 'max': 0.6}
            self.radius_params = {'loc': 1.0, 'scale': 0.4, 'min': 0.4, 'max': 3.0}
            self.mass_params = {'loc': 1.0, 'scale': 0.25, 'min': 0.4, 'max': 2.0}
    
    def sample_stellar_parameters(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Sample realistic stellar parameters.
        
        Args:
            n_samples: Number of stellar parameter sets to sample
            
        Returns:
            Dictionary with stellar parameter arrays
        """
        # Sample from truncated normal distributions
        teff = self._sample_truncated_normal(n_samples, **self.teff_params)
        logg = self._sample_truncated_normal(n_samples, **self.logg_params)
        feh = self._sample_truncated_normal(n_samples, **self.feh_params)
        radius = self._sample_truncated_normal(n_samples, **self.radius_params)
        mass = self._sample_truncated_normal(n_samples, **self.mass_params)
        
        # Apply correlations between parameters
        teff, logg, radius, mass = self._apply_stellar_correlations(teff, logg, radius, mass)
        
        return {
            'teff': teff,
            'logg': logg,
            'feh': feh,
            'radius': radius,
            'mass': mass
        }
    
    def _sample_truncated_normal(
        self, 
        n_samples: int, 
        loc: float, 
        scale: float, 
        min_val: float, 
        max_val: float
    ) -> np.ndarray:
        """Sample from truncated normal distribution."""
        
        # Convert to standard normal bounds
        a = (min_val - loc) / scale
        b = (max_val - loc) / scale
        
        # Sample from truncated normal
        samples = stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=n_samples)
        
        return samples
    
    def _apply_stellar_correlations(
        self, 
        teff: np.ndarray, 
        logg: np.ndarray, 
        radius: np.ndarray, 
        mass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply realistic correlations between stellar parameters."""
        
        # Main sequence correlation: cooler stars are smaller and less massive
        # Adjust radius based on temperature
        teff_norm = (teff - 5778) / 1000  # Normalize to solar
        radius_correction = 1 + 0.3 * teff_norm  # Rough T-R relation
        radius = radius * radius_correction
        
        # Mass-radius correlation for main sequence stars
        # M ∝ R^α where α ≈ 0.8 for main sequence
        mass_correction = radius ** 0.8
        mass = mass * mass_correction
        
        # Surface gravity: log g = log(M/M_sun) - 2*log(R/R_sun) + log(g_sun)
        logg_corrected = np.log10(mass) - 2 * np.log10(radius) + 4.44
        
        # Blend with original logg to avoid over-correction
        logg = 0.7 * logg_corrected + 0.3 * logg
        
        return teff, logg, radius, mass
    
    def get_limb_darkening_coefficients(
        self, 
        teff: Union[float, np.ndarray], 
        logg: Union[float, np.ndarray] = 4.4,
        feh: Union[float, np.ndarray] = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate limb-darkening coefficients based on stellar parameters.
        
        Args:
            teff: Effective temperature (K)
            logg: Surface gravity (log g)
            feh: Metallicity [Fe/H]
            
        Returns:
            Tuple of (u1, u2) limb-darkening coefficients
        """
        # Convert to arrays if needed
        teff = np.atleast_1d(teff)
        logg = np.atleast_1d(logg)
        feh = np.atleast_1d(feh)
        
        # Empirical relations for quadratic limb-darkening in Kepler band
        # Based on Claret & Bloemen (2011) and similar works
        
        # Temperature dependence (primary)
        teff_norm = (teff - 5778) / 1000
        
        # Base coefficients for solar-type star
        u1_base = 0.4
        u2_base = 0.3
        
        # Temperature corrections
        u1 = u1_base + 0.15 * teff_norm - 0.05 * teff_norm**2
        u2 = u2_base - 0.1 * teff_norm + 0.02 * teff_norm**2
        
        # Surface gravity corrections (smaller effect)
        logg_norm = (logg - 4.4) / 0.5
        u1 += 0.02 * logg_norm
        u2 += 0.01 * logg_norm
        
        # Metallicity corrections (small effect)
        u1 += 0.01 * feh
        u2 -= 0.005 * feh
        
        # Ensure physical constraints
        u1 = np.clip(u1, 0.0, 1.0)
        u2 = np.clip(u2, 0.0, 1.0)
        
        # Ensure u1 + u2 <= 1
        total = u1 + u2
        mask = total > 1.0
        if np.any(mask):
            u1[mask] = u1[mask] / total[mask]
            u2[mask] = u2[mask] / total[mask]
        
        return u1, u2


class ExoplanetParameterSampler:
    """
    Samples realistic exoplanet parameters based on observed populations.
    
    Uses empirical distributions from exoplanet surveys to generate
    realistic planetary properties for synthetic transit generation.
    """
    
    def __init__(self, survey_type: str = 'kepler'):
        """
        Initialize exoplanet parameter sampler.
        
        Args:
            survey_type: Type of survey ('kepler', 'tess', 'combined')
        """
        self.survey_type = survey_type
        self._initialize_planet_distributions()
    
    def _initialize_planet_distributions(self):
        """Initialize empirical exoplanet parameter distributions."""
        
        if self.survey_type == 'kepler':
            # Kepler planet population
            self.period_params = {'alpha': 0.5, 'min': 0.5, 'max': 500}  # Power law
            self.radius_params = {'loc': 1.5, 'scale': 1.0, 'min': 0.5, 'max': 10}  # Earth radii
            self.impact_params = {'min': 0.0, 'max': 1.2}  # Uniform in cos(i)
            
        elif self.survey_type == 'tess':
            # TESS focuses on shorter periods
            self.period_params = {'alpha': 0.7, 'min': 0.5, 'max': 50}
            self.radius_params = {'loc': 2.0, 'scale': 1.5, 'min': 0.8, 'max': 15}
            self.impact_params = {'min': 0.0, 'max': 1.0}
            
        else:  # combined
            self.period_params = {'alpha': 0.6, 'min': 0.5, 'max': 300}
            self.radius_params = {'loc': 1.8, 'scale': 1.2, 'min': 0.6, 'max': 12}
            self.impact_params = {'min': 0.0, 'max': 1.1}
    
    def sample_planet_parameters(
        self, 
        stellar_params: Dict[str, np.ndarray],
        n_samples: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Sample realistic planet parameters given stellar properties.
        
        Args:
            stellar_params: Dictionary of stellar parameters
            n_samples: Number of planet parameter sets to sample
            
        Returns:
            Dictionary with planet parameter arrays
        """
        # Sample periods from power law distribution
        periods = self._sample_period_distribution(n_samples)
        
        # Sample planet radii (log-normal distribution)
        planet_radii = self._sample_radius_distribution(n_samples)
        
        # Sample impact parameters (uniform in cos(i) for isotropic orbits)
        impact_parameters = self._sample_impact_parameters(n_samples)
        
        # Calculate transit durations based on stellar and planetary properties
        durations = self._calculate_transit_durations(
            periods, planet_radii, impact_parameters, stellar_params
        )
        
        # Calculate transit depths
        depths = self._calculate_transit_depths(planet_radii, stellar_params)
        
        return {
            'periods': periods,
            'planet_radii_earth': planet_radii,
            'impact_parameters': impact_parameters,
            'durations_hours': durations,
            'depths_ppm': depths
        }
    
    def _sample_period_distribution(self, n_samples: int) -> np.ndarray:
        """Sample orbital periods from power law distribution."""
        
        alpha = self.period_params['alpha']
        p_min = self.period_params['min']
        p_max = self.period_params['max']
        
        # Power law: dN/dP ∝ P^(-alpha)
        # CDF: F(P) = (P^(1-alpha) - P_min^(1-alpha)) / (P_max^(1-alpha) - P_min^(1-alpha))
        
        u = np.random.uniform(0, 1, n_samples)
        
        if alpha == 1:
            # Special case: log-uniform
            periods = p_min * (p_max / p_min) ** u
        else:
            # General power law
            periods = (
                p_min**(1-alpha) + u * (p_max**(1-alpha) - p_min**(1-alpha))
            ) ** (1/(1-alpha))
        
        return periods
    
    def _sample_radius_distribution(self, n_samples: int) -> np.ndarray:
        """Sample planet radii from log-normal distribution."""
        
        loc = self.radius_params['loc']
        scale = self.radius_params['scale']
        min_val = self.radius_params['min']
        max_val = self.radius_params['max']
        
        # Log-normal distribution
        log_radii = np.random.normal(np.log(loc), scale/loc, n_samples)
        radii = np.exp(log_radii)
        
        # Apply bounds
        radii = np.clip(radii, min_val, max_val)
        
        return radii
    
    def _sample_impact_parameters(self, n_samples: int) -> np.ndarray:
        """Sample impact parameters assuming isotropic orbital orientations."""
        
        min_b = self.impact_params['min']
        max_b = self.impact_params['max']
        
        # For isotropic orbits, uniform in cos(i)
        # b = a/R_star * cos(i)
        # For simplicity, sample uniformly in b
        impact_params = np.random.uniform(min_b, max_b, n_samples)
        
        return impact_params
    
    def _calculate_transit_durations(
        self,
        periods: np.ndarray,
        planet_radii: np.ndarray,
        impact_parameters: np.ndarray,
        stellar_params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate transit durations based on orbital and stellar properties."""
        
        # Get stellar radii
        stellar_radii = stellar_params['radius']  # Solar radii
        
        # Ensure arrays are same length
        n_samples = len(periods)
        if len(stellar_radii) == 1:
            stellar_radii = np.full(n_samples, stellar_radii[0])
        
        # Semi-major axis from Kepler's third law (approximate)
        # a^3 = GM*P^2/(4π^2) ≈ M_star * P^2 (in solar units)
        stellar_masses = stellar_params['mass']
        if len(stellar_masses) == 1:
            stellar_masses = np.full(n_samples, stellar_masses[0])
        
        # Semi-major axis in AU
        a_au = (stellar_masses * periods**2)**(1/3)
        
        # Convert to stellar radii (1 AU ≈ 215 R_sun)
        a_rs = a_au * 215 / stellar_radii
        
        # Transit duration (hours)
        # T = P/π * arcsin(R_star/a * sqrt((1+k)^2 - b^2) / sin(i))
        # Simplified: T ≈ P/π * R_star/a * sqrt((1+k)^2 - b^2)
        
        # Planet-to-star radius ratio (approximate)
        k = planet_radii * 0.00916 / stellar_radii  # Earth radii to solar radii
        
        # Duration calculation
        duration_factor = np.sqrt(np.maximum(0, (1 + k)**2 - impact_parameters**2))
        durations = periods * 24 / np.pi * duration_factor / a_rs
        
        # Ensure reasonable bounds
        durations = np.clip(durations, 0.5, 24)  # 0.5 to 24 hours
        
        return durations
    
    def _calculate_transit_depths(
        self,
        planet_radii: np.ndarray,
        stellar_params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calculate transit depths from planet and stellar radii."""
        
        # Get stellar radii
        stellar_radii = stellar_params['radius']  # Solar radii
        
        # Ensure arrays are same length
        n_samples = len(planet_radii)
        if len(stellar_radii) == 1:
            stellar_radii = np.full(n_samples, stellar_radii[0])
        
        # Convert planet radii from Earth radii to solar radii
        planet_radii_solar = planet_radii * 0.00916  # Earth radii to solar radii
        
        # Transit depth = (R_p/R_s)^2
        depths_fraction = (planet_radii_solar / stellar_radii)**2
        
        # Convert to ppm
        depths_ppm = depths_fraction * 1e6
        
        # Ensure reasonable bounds (10 ppm to 50,000 ppm)
        depths_ppm = np.clip(depths_ppm, 10, 50000)
        
        return depths_ppm


class TransitParameterGenerator:
    """
    Combines stellar and planetary parameter sampling to generate complete
    transit parameter sets for synthetic light curve generation.
    """
    
    def __init__(
        self, 
        stellar_catalog: str = 'kepler',
        planet_survey: str = 'kepler'
    ):
        """
        Initialize transit parameter generator.
        
        Args:
            stellar_catalog: Stellar parameter distribution type
            planet_survey: Planet parameter distribution type
        """
        self.stellar_sampler = StellarParameterSampler(stellar_catalog)
        self.planet_sampler = ExoplanetParameterSampler(planet_survey)
    
    def generate_transit_parameters(
        self, 
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> List[TransitParams]:
        """
        Generate complete sets of realistic transit parameters.
        
        Args:
            n_samples: Number of parameter sets to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of TransitParams objects
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample stellar parameters
        stellar_params = self.stellar_sampler.sample_stellar_parameters(n_samples)
        
        # Sample planet parameters
        planet_params = self.planet_sampler.sample_planet_parameters(
            stellar_params, n_samples
        )
        
        # Get limb-darkening coefficients
        u1, u2 = self.stellar_sampler.get_limb_darkening_coefficients(
            stellar_params['teff'], 
            stellar_params['logg'],
            stellar_params['feh']
        )
        
        # Create TransitParams objects
        transit_params_list = []
        
        for i in range(n_samples):
            # Random epoch within first period
            epoch = np.random.uniform(0, planet_params['periods'][i])
            
            params = TransitParams(
                period=float(planet_params['periods'][i]),
                depth=float(planet_params['depths_ppm'][i]),
                duration=float(planet_params['durations_hours'][i]),
                impact_parameter=float(planet_params['impact_parameters'][i]),
                limb_darkening=(float(u1[i]), float(u2[i])),
                epoch=float(epoch)
            )
            
            transit_params_list.append(params)
        
        return transit_params_list
    
    def generate_parameter_statistics(self, n_samples: int = 10000) -> Dict:
        """
        Generate statistics about the parameter distributions.
        
        Args:
            n_samples: Number of samples for statistics
            
        Returns:
            Dictionary with parameter statistics
        """
        # Generate large sample
        params_list = self.generate_transit_parameters(n_samples)
        
        # Extract arrays
        periods = np.array([p.period for p in params_list])
        depths = np.array([p.depth for p in params_list])
        durations = np.array([p.duration for p in params_list])
        impact_params = np.array([p.impact_parameter for p in params_list])
        u1_values = np.array([p.limb_darkening[0] for p in params_list])
        u2_values = np.array([p.limb_darkening[1] for p in params_list])
        
        # Calculate statistics
        stats = {
            'periods': {
                'mean': float(np.mean(periods)),
                'std': float(np.std(periods)),
                'median': float(np.median(periods)),
                'min': float(np.min(periods)),
                'max': float(np.max(periods))
            },
            'depths_ppm': {
                'mean': float(np.mean(depths)),
                'std': float(np.std(depths)),
                'median': float(np.median(depths)),
                'min': float(np.min(depths)),
                'max': float(np.max(depths))
            },
            'durations_hours': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'median': float(np.median(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations))
            },
            'impact_parameters': {
                'mean': float(np.mean(impact_params)),
                'std': float(np.std(impact_params)),
                'median': float(np.median(impact_params)),
                'min': float(np.min(impact_params)),
                'max': float(np.max(impact_params))
            },
            'limb_darkening_u1': {
                'mean': float(np.mean(u1_values)),
                'std': float(np.std(u1_values)),
                'median': float(np.median(u1_values)),
                'min': float(np.min(u1_values)),
                'max': float(np.max(u1_values))
            },
            'limb_darkening_u2': {
                'mean': float(np.mean(u2_values)),
                'std': float(np.std(u2_values)),
                'median': float(np.median(u2_values)),
                'min': float(np.min(u2_values)),
                'max': float(np.max(u2_values))
            }
        }
        
        return stats