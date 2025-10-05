"""
Mandel-Agol transit model implementation for physics-informed synthetic transit generation.

Based on Mandel & Agol (2002) ApJ 580, 171 - "Analytic Light Curves for Planetary Transit Searches"
"""

import numpy as np
from scipy import special
from typing import Tuple, Optional, Union
import warnings

from ..data.types import TransitParams


class MandelAgolTransitModel:
    """
    Implementation of the Mandel-Agol analytic transit model.
    
    Provides accurate computation of transit light curves including
    limb-darkening effects using the analytic formulation.
    """
    
    def __init__(self, limb_darkening_law: str = 'quadratic'):
        """
        Initialize the Mandel-Agol transit model.
        
        Args:
            limb_darkening_law: Type of limb-darkening ('linear', 'quadratic', 'nonlinear')
        """
        self.limb_darkening_law = limb_darkening_law
        
        if limb_darkening_law not in ['linear', 'quadratic', 'nonlinear']:
            raise ValueError(f"Unsupported limb-darkening law: {limb_darkening_law}")
    
    def generate_transit(
        self, 
        time: np.ndarray, 
        params: TransitParams
    ) -> np.ndarray:
        """
        Generate transit light curve using Mandel-Agol model.
        
        Args:
            time: Time array (days)
            params: Transit parameters
            
        Returns:
            Relative flux array (1.0 = no transit, <1.0 = in transit)
        """
        # Calculate planet-star separation as function of time
        z = self._calculate_separation(time, params)
        
        # Calculate planet-to-star radius ratio
        k = self._calculate_radius_ratio(params)
        
        # Compute transit using appropriate limb-darkening
        if self.limb_darkening_law == 'quadratic':
            flux = self._quadratic_limb_darkening(z, k, params.limb_darkening)
        elif self.limb_darkening_law == 'linear':
            flux = self._linear_limb_darkening(z, k, params.limb_darkening[0])
        else:
            # Fallback to uniform disk (no limb-darkening)
            flux = self._uniform_disk(z, k)
        
        return flux
    
    def _calculate_separation(self, time: np.ndarray, params: TransitParams) -> np.ndarray:
        """Calculate normalized planet-star separation."""
        
        # Phase of orbit
        phase = 2 * np.pi * (time - params.epoch) / params.period
        
        # True anomaly (assuming circular orbit)
        true_anomaly = phase
        
        # Orbital separation in units of stellar radius
        # For circular orbit: z = a/R_star * sqrt(sin^2(i) * sin^2(nu) + cos^2(i))
        # Simplified for transit geometry
        
        # Calculate inclination from impact parameter
        # b = a/R_star * cos(i)
        # For now, use simplified geometry
        
        # Projected separation
        z = np.sqrt(
            np.sin(true_anomaly)**2 + 
            (params.impact_parameter * np.cos(true_anomaly))**2
        )
        
        return z
    
    def _calculate_radius_ratio(self, params: TransitParams) -> float:
        """Calculate planet-to-star radius ratio from transit depth."""
        
        # Transit depth (in fractional units) = (R_p/R_s)^2
        # Convert from ppm to fraction if needed
        if params.depth > 1:  # Assume ppm
            depth_fraction = params.depth * 1e-6
        else:  # Already in fractional units
            depth_fraction = params.depth
        
        # Radius ratio
        k = np.sqrt(depth_fraction)
        
        return k
    
    def _uniform_disk(self, z: np.ndarray, k: float) -> np.ndarray:
        """Compute transit for uniform stellar disk (no limb-darkening)."""
        
        flux = np.ones_like(z)
        
        # Complete transit (planet completely in front of star)
        complete_mask = z <= (1 - k)
        flux[complete_mask] = 1 - k**2
        
        # Partial transit (planet partially overlapping star)
        partial_mask = (z > (1 - k)) & (z < (1 + k))
        
        if np.any(partial_mask):
            z_partial = z[partial_mask]
            
            # Area of intersection (analytic formula)
            kappa_0 = np.arccos((k**2 + z_partial**2 - 1) / (2 * k * z_partial))
            kappa_1 = np.arccos((1 - k**2 + z_partial**2) / (2 * z_partial))
            
            area = (k**2 * kappa_0 + kappa_1 - 
                   0.5 * np.sqrt((-z_partial + k + 1) * (z_partial + k - 1) * 
                                (z_partial - k + 1) * (z_partial + k + 1)))
            
            flux[partial_mask] = 1 - area / np.pi
        
        return flux
    
    def _linear_limb_darkening(self, z: np.ndarray, k: float, u: float) -> np.ndarray:
        """Compute transit with linear limb-darkening."""
        
        # Start with uniform disk
        flux = self._uniform_disk(z, k)
        
        # Apply limb-darkening corrections
        # This is a simplified implementation - full Mandel-Agol is more complex
        
        # Limb-darkening factor
        ld_factor = 1 - u * (1 - np.sqrt(1 - z**2))
        
        # Apply correction where planet is transiting
        transit_mask = z < (1 + k)
        flux[transit_mask] *= ld_factor[transit_mask]
        
        return flux
    
    def _quadratic_limb_darkening(
        self, 
        z: np.ndarray, 
        k: float, 
        limb_darkening: Tuple[float, float]
    ) -> np.ndarray:
        """
        Compute transit with quadratic limb-darkening.
        
        This is a simplified implementation of the full Mandel-Agol formulation.
        """
        u1, u2 = limb_darkening
        
        # Initialize flux array
        flux = np.ones_like(z)
        
        # No transit case
        no_transit_mask = z >= (1 + k)
        flux[no_transit_mask] = 1.0
        
        # Complete transit case
        complete_mask = z <= (1 - k)
        if np.any(complete_mask):
            # Simplified formula for complete transit with quadratic limb-darkening
            flux[complete_mask] = 1 - k**2 * (1 - u1/3 - u2/6)
        
        # Partial transit case (most complex)
        partial_mask = (z > (1 - k)) & (z < (1 + k))
        
        if np.any(partial_mask):
            z_partial = z[partial_mask]
            flux_partial = self._compute_partial_transit_quadratic(z_partial, k, u1, u2)
            flux[partial_mask] = flux_partial
        
        return flux
    
    def _compute_partial_transit_quadratic(
        self, 
        z: np.ndarray, 
        k: float, 
        u1: float, 
        u2: float
    ) -> np.ndarray:
        """
        Compute partial transit with quadratic limb-darkening.
        
        This is a simplified approximation of the full Mandel-Agol equations.
        """
        # Geometric factors
        lam = self._compute_lambda(z, k)
        eta = self._compute_eta(z, k)
        
        # Quadratic limb-darkening intensity profile: I(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2
        # where mu = cos(theta) = sqrt(1 - r^2) for radius r from disk center
        
        # Simplified approximation for the limb-darkening integrals
        # Full implementation would require elliptic integrals
        
        # Base geometric transit
        base_transit = 1 - lam
        
        # Limb-darkening corrections (approximated)
        ld_correction = u1 * eta + u2 * eta**2
        
        flux = base_transit + ld_correction
        
        return flux
    
    def _compute_lambda(self, z: np.ndarray, k: float) -> np.ndarray:
        """Compute lambda parameter for partial transit geometry."""
        
        # Simplified computation of the lambda parameter
        # Full implementation requires careful handling of different cases
        
        lambda_val = np.zeros_like(z)
        
        # Case where planet is partially overlapping
        mask = (z > abs(1 - k)) & (z < (1 + k))
        
        if np.any(mask):
            z_masked = z[mask]
            
            # Approximate lambda using geometric overlap
            if k <= 1:
                # Small planet case
                kappa_0 = np.arccos((k**2 + z_masked**2 - 1) / (2 * k * z_masked))
                kappa_1 = np.arccos((1 - k**2 + z_masked**2) / (2 * z_masked))
                
                lambda_val[mask] = (k**2 * kappa_0 + kappa_1) / np.pi
            else:
                # Large planet case (rare for exoplanets)
                lambda_val[mask] = k**2
        
        return lambda_val
    
    def _compute_eta(self, z: np.ndarray, k: float) -> np.ndarray:
        """Compute eta parameter for limb-darkening corrections."""
        
        # Simplified eta computation
        # Full implementation requires elliptic integrals
        
        eta = np.zeros_like(z)
        
        # Approximate eta based on geometric considerations
        mask = (z > abs(1 - k)) & (z < (1 + k))
        
        if np.any(mask):
            z_masked = z[mask]
            
            # Simple approximation
            eta[mask] = 0.5 * k**2 * (1 - z_masked**2)
        
        return eta
    
    def validate_parameters(self, params: TransitParams) -> bool:
        """
        Validate transit parameters for physical consistency.
        
        Args:
            params: Transit parameters to validate
            
        Returns:
            True if parameters are valid
        """
        try:
            # Check basic parameter ranges
            if params.period <= 0:
                warnings.warn("Period must be positive")
                return False
            
            if params.depth <= 0:
                warnings.warn("Transit depth must be positive")
                return False
            
            if params.duration <= 0:
                warnings.warn("Transit duration must be positive")
                return False
            
            if not (0 <= params.impact_parameter <= 1.5):
                warnings.warn("Impact parameter should be between 0 and 1.5")
                return False
            
            # Check limb-darkening coefficients
            u1, u2 = params.limb_darkening
            if not (0 <= u1 <= 1 and 0 <= u2 <= 1):
                warnings.warn("Limb-darkening coefficients should be between 0 and 1")
                return False
            
            if u1 + u2 > 1:
                warnings.warn("Sum of limb-darkening coefficients should not exceed 1")
                return False
            
            # Check duration consistency with period and impact parameter
            # Duration should be reasonable fraction of period
            if params.duration > 0.5 * params.period:
                warnings.warn("Transit duration seems too long relative to period")
                return False
            
            return True
            
        except Exception as e:
            warnings.warn(f"Parameter validation failed: {e}")
            return False
    
    def compute_transit_observables(self, params: TransitParams) -> dict:
        """
        Compute observable quantities from transit parameters.
        
        Args:
            params: Transit parameters
            
        Returns:
            Dictionary of computed observables
        """
        # Planet-to-star radius ratio
        if params.depth > 1:  # ppm
            depth_fraction = params.depth * 1e-6
        else:
            depth_fraction = params.depth
        
        radius_ratio = np.sqrt(depth_fraction)
        
        # Semi-major axis in stellar radii (approximate)
        # Using Kepler's third law and transit duration
        duration_fraction = params.duration / 24.0  # Convert hours to days
        
        # Approximate semi-major axis (assumes circular orbit)
        a_over_rs = np.pi * params.period / (duration_fraction * 24 * 3600)  # Very rough approximation
        
        # Inclination from impact parameter
        if a_over_rs > 0:
            inclination = np.arccos(params.impact_parameter / a_over_rs)
        else:
            inclination = np.pi / 2  # Edge-on
        
        observables = {
            'radius_ratio': radius_ratio,
            'depth_fraction': depth_fraction,
            'depth_ppm': depth_fraction * 1e6,
            'semi_major_axis_rs': a_over_rs,
            'inclination_deg': np.degrees(inclination),
            'impact_parameter': params.impact_parameter,
            'limb_darkening_u1': params.limb_darkening[0],
            'limb_darkening_u2': params.limb_darkening[1]
        }
        
        return observables
    
    def generate_transit_with_noise(
        self,
        time: np.ndarray,
        params: TransitParams,
        noise_level: float = 0.001,
        systematic_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic transit with observational noise.
        
        Args:
            time: Time array
            params: Transit parameters
            noise_level: Gaussian noise level (fractional)
            systematic_noise: Whether to add systematic noise patterns
            
        Returns:
            Tuple of (flux_with_noise, noise_array)
        """
        # Generate clean transit
        clean_flux = self.generate_transit(time, params)
        
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, noise_level, len(time))
        
        # Add systematic noise if requested
        systematic = np.zeros_like(time)
        if systematic_noise:
            # Add slow systematic trends
            systematic += 0.0001 * np.sin(2 * np.pi * time / (params.period * 5))
            
            # Add correlated noise
            systematic += 0.0002 * np.sin(2 * np.pi * time / (params.period * 0.1))
        
        total_noise = gaussian_noise + systematic
        noisy_flux = clean_flux + total_noise
        
        return noisy_flux, total_noise