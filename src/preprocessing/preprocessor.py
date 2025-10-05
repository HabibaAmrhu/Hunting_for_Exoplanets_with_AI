"""
Light curve preprocessing module for standardization, detrending, and normalization.
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.ndimage import median_filter
from typing import Optional, Tuple, Union
import warnings

from ..data.types import LightCurve, ProcessedLightCurve, PreprocessingConfig
import torch


class LightCurvePreprocessor:
    """
    Comprehensive light curve preprocessing for exoplanet detection.
    
    Handles standardization, detrending, normalization, and missing value
    interpolation to prepare light curves for machine learning models.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration parameters
        """
        self.config = config or PreprocessingConfig()
        
    def process(self, light_curve: LightCurve) -> ProcessedLightCurve:
        """
        Apply full preprocessing pipeline to a light curve.
        
        Args:
            light_curve: Raw light curve data
            
        Returns:
            Processed light curve ready for model input
        """
        # Step 1: Handle missing values
        time, flux, flux_err = self._handle_missing_values(
            light_curve.time, light_curve.flux, light_curve.flux_err
        )
        
        # Step 2: Detrend the light curve
        detrended_flux = self._detrend(time, flux)
        
        # Step 3: Normalize the flux
        normalized_flux = self._normalize(detrended_flux)
        
        # Step 4: Standardize length
        standardized_time, standardized_flux = self._standardize_length(
            time, normalized_flux
        )
        
        # Step 5: Create phase-folded version if period is available
        phase_folded_flux = self._create_phase_folded(
            standardized_time, standardized_flux, light_curve.period
        )
        
        # Step 6: Create mask for missing/interpolated data
        mask = self._create_mask(standardized_flux)
        
        # Step 7: Convert to tensors
        raw_tensor = torch.from_numpy(standardized_flux).float()
        phase_tensor = torch.from_numpy(phase_folded_flux).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        # Calculate confidence weight based on data quality
        confidence_weight = self._calculate_confidence_weight(
            light_curve, standardized_flux
        )
        
        return ProcessedLightCurve(
            raw_flux=raw_tensor,
            phase_folded_flux=phase_tensor,
            mask=mask_tensor,
            label=light_curve.label,
            confidence_weight=confidence_weight,
            augmentation_params={}
        )
    
    def _handle_missing_values(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        flux_err: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle missing values through interpolation or removal."""
        
        # Find finite values
        finite_mask = np.isfinite(flux) & np.isfinite(time) & np.isfinite(flux_err)
        
        if not np.any(finite_mask):
            raise ValueError("No finite values in light curve")
        
        # If too many missing values, raise warning
        missing_fraction = 1 - np.sum(finite_mask) / len(finite_mask)
        if missing_fraction > 0.5:
            warnings.warn(f"High fraction of missing data: {missing_fraction:.2%}")
        
        # Extract finite values
        clean_time = time[finite_mask]
        clean_flux = flux[finite_mask]
        clean_flux_err = flux_err[finite_mask]
        
        # Interpolate small gaps if requested
        if self.config.interpolation_method and missing_fraction < 0.2:
            clean_time, clean_flux, clean_flux_err = self._interpolate_gaps(
                time, flux, flux_err, finite_mask
            )
        
        return clean_time, clean_flux, clean_flux_err
    
    def _interpolate_gaps(
        self,
        time: np.ndarray,
        flux: np.ndarray, 
        flux_err: np.ndarray,
        finite_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate small gaps in the light curve."""
        
        if self.config.interpolation_method == 'linear':
            # Linear interpolation for flux
            interp_func = interpolate.interp1d(
                time[finite_mask], 
                flux[finite_mask], 
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Interpolate flux errors (use median for simplicity)
            median_err = np.median(flux_err[finite_mask])
            
            interpolated_flux = interp_func(time)
            interpolated_flux_err = np.full_like(time, median_err)
            
            # Only interpolate small gaps (< 5% of total duration)
            gap_threshold = 0.05 * (np.max(time) - np.min(time))
            
            # Find gaps
            time_diffs = np.diff(time[finite_mask])
            large_gaps = time_diffs > gap_threshold
            
            if np.any(large_gaps):
                # Don't interpolate across large gaps
                gap_indices = np.where(large_gaps)[0]
                for gap_idx in gap_indices:
                    start_idx = np.where(time >= time[finite_mask][gap_idx])[0]
                    end_idx = np.where(time <= time[finite_mask][gap_idx + 1])[0]
                    
                    if len(start_idx) > 0 and len(end_idx) > 0:
                        gap_mask = (time >= time[start_idx[0]]) & (time <= time[end_idx[-1]])
                        interpolated_flux[gap_mask] = np.nan
            
            # Remove NaN values after interpolation
            valid_mask = np.isfinite(interpolated_flux)
            return time[valid_mask], interpolated_flux[valid_mask], interpolated_flux_err[valid_mask]
        
        else:
            # Just return finite values
            return time[finite_mask], flux[finite_mask], flux_err[finite_mask]
    
    def _detrend(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Remove long-term trends from the light curve."""
        
        if self.config.detrend_method == 'median':
            return self._detrend_median_filter(flux)
        elif self.config.detrend_method == 'savgol':
            return self._detrend_savitzky_golay(flux)
        else:
            raise ValueError(f"Unknown detrend method: {self.config.detrend_method}")
    
    def _detrend_median_filter(self, flux: np.ndarray) -> np.ndarray:
        """Detrend using median filter."""
        
        # Apply median filter to estimate trend
        filter_size = min(self.config.median_filter_size, len(flux) // 4)
        if filter_size % 2 == 0:
            filter_size += 1  # Ensure odd size
        
        if filter_size < 3:
            # Too short for meaningful detrending
            return flux - np.median(flux)
        
        trend = median_filter(flux, size=filter_size, mode='reflect')
        
        # Remove trend
        detrended = flux - trend + np.median(flux)
        
        return detrended
    
    def _detrend_savitzky_golay(self, flux: np.ndarray) -> np.ndarray:
        """Detrend using Savitzky-Golay filter."""
        
        window_length = min(self.config.savgol_window, len(flux))
        if window_length % 2 == 0:
            window_length -= 1  # Ensure odd
        
        if window_length < self.config.savgol_polyorder + 2:
            # Fallback to median filter
            return self._detrend_median_filter(flux)
        
        try:
            # Apply Savitzky-Golay filter to estimate trend
            trend = signal.savgol_filter(
                flux, 
                window_length, 
                self.config.savgol_polyorder,
                mode='nearest'
            )
            
            # Remove trend
            detrended = flux - trend + np.median(flux)
            
            return detrended
            
        except Exception:
            # Fallback to median filter if Savitzky-Golay fails
            return self._detrend_median_filter(flux)
    
    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        """Normalize the flux values."""
        
        if self.config.normalization == 'zscore':
            # Z-score normalization (subtract median, divide by std)
            median_flux = np.median(flux)
            std_flux = np.std(flux)
            
            if std_flux == 0:
                return flux - median_flux
            
            return (flux - median_flux) / std_flux
            
        elif self.config.normalization == 'minmax':
            # Min-max normalization to [0, 1]
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            
            if max_flux == min_flux:
                return np.zeros_like(flux)
            
            return (flux - min_flux) / (max_flux - min_flux)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization}")
    
    def _standardize_length(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize light curve to target length."""
        
        target_length = self.config.target_length
        current_length = len(flux)
        
        if current_length == target_length:
            return time, flux
        
        elif current_length > target_length:
            # Downsample using interpolation for smooth result
            return self._downsample(time, flux, target_length)
            
        else:
            # Upsample using interpolation
            return self._upsample(time, flux, target_length)
    
    def _downsample(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample light curve to target length."""
        
        # Create evenly spaced time grid
        new_time = np.linspace(time[0], time[-1], target_length)
        
        # Interpolate flux to new time grid
        interp_func = interpolate.interp1d(
            time, flux, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        new_flux = interp_func(new_time)
        
        return new_time, new_flux
    
    def _upsample(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Upsample light curve to target length."""
        
        # Create evenly spaced time grid
        new_time = np.linspace(time[0], time[-1], target_length)
        
        # Interpolate flux to new time grid
        interp_func = interpolate.interp1d(
            time, flux, kind='cubic' if len(time) > 3 else 'linear',
            bounds_error=False, fill_value='extrapolate'
        )
        new_flux = interp_func(new_time)
        
        return new_time, new_flux
    
    def _create_phase_folded(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        period: Optional[float]
    ) -> np.ndarray:
        """Create phase-folded version of the light curve."""
        
        if period is None or period <= 0:
            # No period available, return copy of original flux
            return flux.copy()
        
        # Calculate phases
        phases = (time % period) / period
        
        # Sort by phase
        sort_indices = np.argsort(phases)
        sorted_phases = phases[sort_indices]
        sorted_flux = flux[sort_indices]
        
        # Create evenly spaced phase grid
        phase_grid = np.linspace(0, 1, len(flux))
        
        # Interpolate to even phase spacing
        # Handle phase wrapping by duplicating data at boundaries
        extended_phases = np.concatenate([
            sorted_phases - 1,  # Previous cycle
            sorted_phases,      # Current cycle  
            sorted_phases + 1   # Next cycle
        ])
        extended_flux = np.concatenate([sorted_flux, sorted_flux, sorted_flux])
        
        # Interpolate
        interp_func = interpolate.interp1d(
            extended_phases, extended_flux, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        
        phase_folded_flux = interp_func(phase_grid)
        
        return phase_folded_flux
    
    def _create_mask(self, flux: np.ndarray) -> np.ndarray:
        """Create mask indicating valid data points."""
        
        # Mark finite values as valid (1), others as invalid (0)
        mask = np.isfinite(flux).astype(np.float32)
        
        return mask
    
    def _calculate_confidence_weight(
        self, 
        light_curve: LightCurve, 
        processed_flux: np.ndarray
    ) -> float:
        """Calculate confidence weight based on data quality."""
        
        # Base weight
        weight = 1.0
        
        # Adjust based on light curve length
        original_length = len(light_curve.flux)
        if original_length < 500:
            weight *= 0.8  # Lower confidence for short light curves
        elif original_length > 2000:
            weight *= 1.2  # Higher confidence for long light curves
        
        # Adjust based on noise level
        if hasattr(light_curve, 'flux_err') and light_curve.flux_err is not None:
            median_snr = np.median(np.abs(light_curve.flux) / light_curve.flux_err)
            if median_snr < 10:
                weight *= 0.9  # Lower confidence for noisy data
            elif median_snr > 50:
                weight *= 1.1  # Higher confidence for clean data
        
        # Adjust based on variability
        flux_std = np.std(processed_flux)
        if flux_std > 3:  # Very variable
            weight *= 0.9
        elif flux_std < 0.5:  # Very stable
            weight *= 1.1
        
        # Clamp weight to reasonable range
        weight = np.clip(weight, 0.5, 2.0)
        
        return float(weight)
    
    def batch_process(
        self, 
        light_curves: list[LightCurve],
        show_progress: bool = True
    ) -> list[ProcessedLightCurve]:
        """Process multiple light curves in batch."""
        
        processed_curves = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(light_curves, desc="Processing light curves")
            except ImportError:
                iterator = light_curves
        else:
            iterator = light_curves
        
        for lc in iterator:
            try:
                processed = self.process(lc)
                processed_curves.append(processed)
            except Exception as e:
                warnings.warn(f"Failed to process {lc.star_id}: {e}")
                continue
        
        return processed_curves
    
    def get_preprocessing_stats(
        self, 
        original: LightCurve, 
        processed: ProcessedLightCurve
    ) -> dict:
        """Get statistics about the preprocessing operation."""
        
        stats = {
            'original_length': len(original.flux),
            'processed_length': len(processed.raw_flux),
            'original_std': float(np.std(original.flux)),
            'processed_std': float(torch.std(processed.raw_flux).item()),
            'confidence_weight': processed.confidence_weight,
            'has_period': original.period is not None,
            'period_value': original.period,
            'missing_data_fraction': 1.0 - float(torch.mean(processed.mask).item())
        }
        
        return stats