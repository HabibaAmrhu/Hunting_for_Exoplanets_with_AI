"""
Advanced phase-folding functionality for exoplanet detection.
"""

import numpy as np
from scipy import interpolate, optimize
from typing import Optional, Tuple, Dict, List
import warnings

from ..data.types import LightCurve


class PhaseFoldingEngine:
    """
    Advanced phase-folding engine with period detection and optimization.
    
    Provides sophisticated phase-folding capabilities including automatic
    period detection, phase optimization, and multi-period analysis.
    """
    
    def __init__(self):
        """Initialize the phase-folding engine."""
        pass
    
    def fold_light_curve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        epoch: Optional[float] = None,
        phase_bins: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fold light curve to given period with optional epoch adjustment.
        
        Args:
            time: Time array
            flux: Flux array
            period: Folding period
            epoch: Reference epoch (time of transit center)
            phase_bins: Number of phase bins for output
            
        Returns:
            Tuple of (phase_array, folded_flux_array)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        # Calculate phases
        if epoch is not None:
            phases = ((time - epoch) % period) / period
        else:
            phases = (time % period) / period
        
        # Create phase grid
        phase_grid = np.linspace(0, 1, phase_bins)
        
        # Bin and average the data
        folded_flux = self._bin_phase_data(phases, flux, phase_grid)
        
        return phase_grid, folded_flux
    
    def _bin_phase_data(
        self,
        phases: np.ndarray,
        flux: np.ndarray,
        phase_grid: np.ndarray
    ) -> np.ndarray:
        """Bin phase data onto regular grid."""
        
        # Sort by phase
        sort_indices = np.argsort(phases)
        sorted_phases = phases[sort_indices]
        sorted_flux = flux[sort_indices]
        
        # Handle phase wrapping by extending data
        extended_phases = np.concatenate([
            sorted_phases - 1,  # Previous cycle
            sorted_phases,      # Current cycle
            sorted_phases + 1   # Next cycle
        ])
        extended_flux = np.concatenate([sorted_flux, sorted_flux, sorted_flux])
        
        # Interpolate to regular phase grid
        interp_func = interpolate.interp1d(
            extended_phases,
            extended_flux,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        folded_flux = interp_func(phase_grid)
        
        return folded_flux
    
    def optimize_epoch(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        initial_epoch: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Optimize epoch to maximize transit signal.
        
        Args:
            time: Time array
            flux: Flux array  
            period: Known period
            initial_epoch: Initial guess for epoch
            
        Returns:
            Tuple of (optimized_epoch, signal_strength)
        """
        if initial_epoch is None:
            initial_epoch = time[0]
        
        def objective(epoch):
            """Objective function to minimize (negative signal strength)."""
            phases = ((time - epoch) % period) / period
            
            # Calculate signal strength as depth of central transit
            central_mask = np.abs(phases - 0.5) < 0.1  # Central 20% of phase
            if np.sum(central_mask) < 5:
                return 0  # Not enough points
            
            central_flux = flux[central_mask]
            baseline_flux = flux[~central_mask]
            
            if len(baseline_flux) == 0:
                return 0
            
            signal_strength = np.median(baseline_flux) - np.median(central_flux)
            return -signal_strength  # Minimize negative signal
        
        # Optimize epoch
        result = optimize.minimize_scalar(
            objective,
            bounds=(time[0], time[0] + period),
            method='bounded'
        )
        
        optimized_epoch = result.x
        signal_strength = -result.fun
        
        return optimized_epoch, signal_strength
    
    def detect_period_bls(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period_min: float = 0.5,
        period_max: float = 50.0,
        frequency_factor: float = 5.0
    ) -> Tuple[float, float, Dict]:
        """
        Detect period using Box Least Squares (BLS) algorithm.
        
        Args:
            time: Time array
            flux: Flux array
            period_min: Minimum period to search
            period_max: Maximum period to search
            frequency_factor: Frequency oversampling factor
            
        Returns:
            Tuple of (best_period, bls_power, bls_results)
        """
        try:
            from astropy.timeseries import BoxLeastSquares
            
            # Create BLS object
            bls = BoxLeastSquares(time, flux)
            
            # Define frequency grid
            duration = time[-1] - time[0]
            frequency_min = 1.0 / period_max
            frequency_max = 1.0 / period_min
            
            # Calculate number of frequencies
            n_freq = int(frequency_factor * duration * (frequency_max - frequency_min))
            frequencies = np.linspace(frequency_min, frequency_max, n_freq)
            
            # Run BLS
            periodogram = bls.power(frequencies, duration_fraction=0.1)
            
            # Find best period
            best_idx = np.argmax(periodogram.power)
            best_period = 1.0 / frequencies[best_idx]
            best_power = periodogram.power[best_idx]
            
            results = {
                'periods': 1.0 / frequencies,
                'power': periodogram.power,
                'frequencies': frequencies,
                'best_index': best_idx
            }
            
            return best_period, best_power, results
            
        except ImportError:
            warnings.warn("Astropy not available, using simple periodogram")
            return self._simple_periodogram(time, flux, period_min, period_max)
    
    def _simple_periodogram(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period_min: float,
        period_max: float
    ) -> Tuple[float, float, Dict]:
        """Simple periodogram implementation as fallback."""
        
        # Create period grid
        n_periods = 1000
        periods = np.linspace(period_min, period_max, n_periods)
        power = np.zeros(n_periods)
        
        # Calculate power for each period
        for i, period in enumerate(periods):
            phases = (time % period) / period
            
            # Simple box model fit
            phase_bins = np.linspace(0, 1, 50)
            binned_flux = []
            
            for j in range(len(phase_bins) - 1):
                mask = (phases >= phase_bins[j]) & (phases < phase_bins[j + 1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(flux[mask]))
                else:
                    binned_flux.append(np.median(flux))
            
            binned_flux = np.array(binned_flux)
            
            # Calculate power as variance of binned flux
            power[i] = np.var(binned_flux)
        
        # Find best period
        best_idx = np.argmax(power)
        best_period = periods[best_idx]
        best_power = power[best_idx]
        
        results = {
            'periods': periods,
            'power': power,
            'best_index': best_idx
        }
        
        return best_period, best_power, results
    
    def create_dual_channel_input(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: Optional[float] = None,
        target_length: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dual-channel input (raw + phase-folded) for neural networks.
        
        Args:
            time: Time array
            flux: Flux array
            period: Folding period (if None, uses raw flux for both channels)
            target_length: Target length for both channels
            
        Returns:
            Tuple of (raw_channel, phase_folded_channel)
        """
        # Standardize raw flux to target length
        raw_channel = self._standardize_to_length(time, flux, target_length)
        
        if period is not None and period > 0:
            # Create phase-folded channel
            phase_grid, folded_flux = self.fold_light_curve(
                time, flux, period, phase_bins=target_length
            )
            phase_folded_channel = folded_flux
        else:
            # No period available, use raw flux for both channels
            phase_folded_channel = raw_channel.copy()
        
        return raw_channel, phase_folded_channel
    
    def _standardize_to_length(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """Standardize flux array to target length."""
        
        if len(flux) == target_length:
            return flux
        
        # Create evenly spaced time grid
        new_time = np.linspace(time[0], time[-1], target_length)
        
        # Interpolate flux
        interp_func = interpolate.interp1d(
            time, flux, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        return interp_func(new_time)
    
    def analyze_phase_coverage(
        self,
        time: np.ndarray,
        period: float,
        epoch: Optional[float] = None
    ) -> Dict:
        """
        Analyze phase coverage for a given period.
        
        Args:
            time: Time array
            period: Period to analyze
            epoch: Reference epoch
            
        Returns:
            Dictionary with coverage statistics
        """
        if epoch is None:
            epoch = time[0]
        
        phases = ((time - epoch) % period) / period
        
        # Calculate phase coverage statistics
        phase_bins = np.linspace(0, 1, 100)
        hist, _ = np.histogram(phases, bins=phase_bins)
        
        coverage_fraction = np.sum(hist > 0) / len(hist)
        max_gap = self._find_max_phase_gap(phases)
        
        # Transit coverage (assuming transit at phase 0.5)
        transit_phases = phases[(phases > 0.4) & (phases < 0.6)]
        transit_coverage = len(transit_phases) / len(phases)
        
        return {
            'coverage_fraction': coverage_fraction,
            'max_phase_gap': max_gap,
            'transit_coverage': transit_coverage,
            'n_transits': len(time) * period / (time[-1] - time[0]),
            'phase_distribution': hist
        }
    
    def _find_max_phase_gap(self, phases: np.ndarray) -> float:
        """Find maximum gap in phase coverage."""
        
        sorted_phases = np.sort(phases)
        
        # Calculate gaps between consecutive phases
        gaps = np.diff(sorted_phases)
        
        # Also check wrap-around gap
        wrap_gap = 1.0 - sorted_phases[-1] + sorted_phases[0]
        
        all_gaps = np.append(gaps, wrap_gap)
        
        return np.max(all_gaps)
    
    def multi_period_analysis(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        periods: List[float]
    ) -> Dict:
        """
        Analyze multiple periods and compare their phase-folded signals.
        
        Args:
            time: Time array
            flux: Flux array
            periods: List of periods to analyze
            
        Returns:
            Dictionary with analysis results for each period
        """
        results = {}
        
        for period in periods:
            # Fold at this period
            phase_grid, folded_flux = self.fold_light_curve(time, flux, period)
            
            # Optimize epoch
            opt_epoch, signal_strength = self.optimize_epoch(time, flux, period)
            
            # Analyze coverage
            coverage = self.analyze_phase_coverage(time, period, opt_epoch)
            
            # Calculate signal metrics
            signal_metrics = self._calculate_signal_metrics(folded_flux)
            
            results[period] = {
                'folded_flux': folded_flux,
                'phase_grid': phase_grid,
                'optimized_epoch': opt_epoch,
                'signal_strength': signal_strength,
                'coverage': coverage,
                'metrics': signal_metrics
            }
        
        return results
    
    def _calculate_signal_metrics(self, folded_flux: np.ndarray) -> Dict:
        """Calculate signal quality metrics for folded light curve."""
        
        # Transit depth (assuming transit at phase 0.5)
        n_points = len(folded_flux)
        center_idx = n_points // 2
        transit_width = n_points // 20  # 5% of phase
        
        transit_region = folded_flux[center_idx - transit_width:center_idx + transit_width]
        baseline_region = np.concatenate([
            folded_flux[:center_idx - 2*transit_width],
            folded_flux[center_idx + 2*transit_width:]
        ])
        
        if len(baseline_region) == 0:
            baseline_flux = np.median(folded_flux)
        else:
            baseline_flux = np.median(baseline_region)
        
        transit_depth = baseline_flux - np.median(transit_region)
        
        # Signal-to-noise ratio
        noise_level = np.std(baseline_region) if len(baseline_region) > 0 else np.std(folded_flux)
        snr = transit_depth / noise_level if noise_level > 0 else 0
        
        # Symmetry measure
        left_half = folded_flux[:n_points//2]
        right_half = folded_flux[n_points//2:][::-1]  # Reverse right half
        
        if len(left_half) == len(right_half):
            symmetry = np.corrcoef(left_half, right_half)[0, 1]
        else:
            symmetry = 0
        
        return {
            'transit_depth': transit_depth,
            'signal_to_noise': snr,
            'symmetry': symmetry,
            'baseline_flux': baseline_flux,
            'noise_level': noise_level
        }