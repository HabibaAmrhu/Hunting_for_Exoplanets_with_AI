"""Data augmentation pipeline for exoplanet detection."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import random
from scipy import signal
from scipy.interpolate import interp1d

from .types import LightCurve


class BaseAugmentation(ABC):
    """Base class for all augmentation techniques."""
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentation.
        
        Args:
            probability: Probability of applying this augmentation
        """
        self.probability = probability
    
    @abstractmethod
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply augmentation to light curve.
        
        Args:
            light_curve: Input light curve data
            metadata: Optional metadata dictionary
            
        Returns:
            Tuple of (augmented_light_curve, updated_metadata)
        """
        pass
    
    def should_apply(self) -> bool:
        """Determine if augmentation should be applied based on probability."""
        return random.random() < self.probability


class TimeJitter(BaseAugmentation):
    """Apply random time shifts to the light curve."""
    
    def __init__(self, max_shift: int = 50, probability: float = 0.5):
        """
        Initialize time jitter augmentation.
        
        Args:
            max_shift: Maximum number of time steps to shift
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.max_shift = max_shift
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply time jitter to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        # Generate random shift
        shift = random.randint(-self.max_shift, self.max_shift)
        
        # Apply shift with padding
        if shift > 0:
            # Shift right, pad left
            augmented = np.pad(light_curve, ((0, 0), (shift, 0)), mode='edge')[:, :-shift]
        elif shift < 0:
            # Shift left, pad right
            augmented = np.pad(light_curve, ((0, 0), (0, -shift)), mode='edge')[:, -shift:]
        else:
            augmented = light_curve.copy()
        
        metadata['time_jitter_shift'] = shift
        return augmented, metadata


class AmplitudeScaling(BaseAugmentation):
    """Apply random amplitude scaling to the light curve."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.5):
        """
        Initialize amplitude scaling augmentation.
        
        Args:
            scale_range: Range of scaling factors (min, max)
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.scale_range = scale_range
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply amplitude scaling to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        # Generate random scale factor
        scale_factor = random.uniform(*self.scale_range)
        
        # Apply scaling
        augmented = light_curve * scale_factor
        
        metadata['amplitude_scale_factor'] = scale_factor
        return augmented, metadata


class GaussianNoise(BaseAugmentation):
    """Add Gaussian noise to the light curve."""
    
    def __init__(self, noise_std: float = 0.01, probability: float = 0.5):
        """
        Initialize Gaussian noise augmentation.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.noise_std = noise_std
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Add Gaussian noise to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        # Generate noise
        noise = np.random.normal(0, self.noise_std, light_curve.shape)
        
        # Add noise
        augmented = light_curve + noise
        
        metadata['gaussian_noise_std'] = self.noise_std
        return augmented, metadata


class RandomMasking(BaseAugmentation):
    """Apply random masking to simulate data gaps."""
    
    def __init__(self, mask_fraction: float = 0.05, max_mask_length: int = 20, probability: float = 0.3):
        """
        Initialize random masking augmentation.
        
        Args:
            mask_fraction: Fraction of data to mask
            max_mask_length: Maximum length of continuous mask
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.mask_fraction = mask_fraction
        self.max_mask_length = max_mask_length
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply random masking to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        augmented = light_curve.copy()
        sequence_length = light_curve.shape[-1]
        
        # Calculate total points to mask
        total_mask_points = int(sequence_length * self.mask_fraction)
        masked_points = 0
        mask_positions = []
        
        while masked_points < total_mask_points:
            # Random start position
            start_pos = random.randint(0, sequence_length - 1)
            
            # Random mask length
            remaining_points = total_mask_points - masked_points
            max_length = min(self.max_mask_length, remaining_points, sequence_length - start_pos)
            mask_length = random.randint(1, max_length)
            
            end_pos = start_pos + mask_length
            
            # Apply mask (interpolate between boundaries)
            if start_pos > 0 and end_pos < sequence_length:
                # Linear interpolation
                for channel in range(augmented.shape[0]):
                    start_val = augmented[channel, start_pos - 1]
                    end_val = augmented[channel, end_pos]
                    interpolated = np.linspace(start_val, end_val, mask_length + 2)[1:-1]
                    augmented[channel, start_pos:end_pos] = interpolated
            else:
                # Edge case: use nearest neighbor
                for channel in range(augmented.shape[0]):
                    if start_pos == 0:
                        augmented[channel, start_pos:end_pos] = augmented[channel, end_pos]
                    else:
                        augmented[channel, start_pos:end_pos] = augmented[channel, start_pos - 1]
            
            mask_positions.append((start_pos, end_pos))
            masked_points += mask_length
        
        metadata['mask_positions'] = mask_positions
        metadata['mask_fraction_actual'] = masked_points / sequence_length
        return augmented, metadata


class FrequencyFiltering(BaseAugmentation):
    """Apply frequency domain filtering."""
    
    def __init__(self, filter_type: str = 'lowpass', cutoff_range: Tuple[float, float] = (0.1, 0.4), 
                 probability: float = 0.3):
        """
        Initialize frequency filtering augmentation.
        
        Args:
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            cutoff_range: Range of cutoff frequencies (normalized)
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.filter_type = filter_type
        self.cutoff_range = cutoff_range
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply frequency filtering to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        augmented = light_curve.copy()
        
        # Generate random cutoff frequency
        cutoff = random.uniform(*self.cutoff_range)
        
        # Apply filter to each channel
        for channel in range(augmented.shape[0]):
            if self.filter_type == 'lowpass':
                b, a = signal.butter(4, cutoff, btype='low')
            elif self.filter_type == 'highpass':
                b, a = signal.butter(4, cutoff, btype='high')
            elif self.filter_type == 'bandpass':
                low_cutoff = cutoff
                high_cutoff = min(cutoff + 0.2, 0.5)
                b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            else:
                continue
            
            # Apply filter
            augmented[channel] = signal.filtfilt(b, a, augmented[channel])
        
        metadata['frequency_filter'] = {
            'type': self.filter_type,
            'cutoff': cutoff
        }
        return augmented, metadata


class TimeWarping(BaseAugmentation):
    """Apply time warping to simulate timing variations."""
    
    def __init__(self, warp_strength: float = 0.1, probability: float = 0.2):
        """
        Initialize time warping augmentation.
        
        Args:
            warp_strength: Strength of time warping
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.warp_strength = warp_strength
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply time warping to light curve."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        sequence_length = light_curve.shape[-1]
        
        # Generate warping function
        n_knots = 5
        knot_positions = np.linspace(0, sequence_length - 1, n_knots)
        warp_offsets = np.random.normal(0, self.warp_strength * sequence_length / n_knots, n_knots)
        
        # Ensure monotonic warping
        warp_offsets = np.cumsum(warp_offsets)
        warp_offsets = warp_offsets - warp_offsets[0]  # Start at 0
        
        warped_positions = knot_positions + warp_offsets
        warped_positions = np.clip(warped_positions, 0, sequence_length - 1)
        
        # Create interpolation function
        original_indices = np.arange(sequence_length)
        interp_func = interp1d(warped_positions, knot_positions, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
        new_indices = interp_func(original_indices)
        new_indices = np.clip(new_indices, 0, sequence_length - 1)
        
        # Apply warping to each channel
        augmented = np.zeros_like(light_curve)
        for channel in range(light_curve.shape[0]):
            interp_func = interp1d(original_indices, light_curve[channel], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            augmented[channel] = interp_func(new_indices)
        
        metadata['time_warp_strength'] = self.warp_strength
        return augmented, metadata


class MixUp(BaseAugmentation):
    """Apply MixUp augmentation between samples."""
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.3):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.alpha = alpha
        self._mix_samples = []
    
    def set_mix_samples(self, samples: List[Tuple[np.ndarray, int]]):
        """Set samples to mix with."""
        self._mix_samples = samples
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply MixUp augmentation."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply() or not self._mix_samples:
            return light_curve, metadata
        
        # Select random sample to mix with
        mix_sample, mix_label = random.choice(self._mix_samples)
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix samples
        augmented = lam * light_curve + (1 - lam) * mix_sample
        
        metadata['mixup'] = {
            'lambda': lam,
            'mixed_with_label': mix_label
        }
        
        return augmented, metadata


class TransitPreservingAugmentation(BaseAugmentation):
    """Augmentation that preserves transit signals while modifying background."""
    
    def __init__(self, transit_mask: Optional[np.ndarray] = None, probability: float = 0.4):
        """
        Initialize transit-preserving augmentation.
        
        Args:
            transit_mask: Boolean mask indicating transit regions
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.transit_mask = transit_mask
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply transit-preserving augmentation."""
        if metadata is None:
            metadata = {}
        
        if not self.should_apply():
            return light_curve, metadata
        
        augmented = light_curve.copy()
        
        # If no transit mask provided, try to detect transit automatically
        if self.transit_mask is None:
            # Simple transit detection: find significant dips
            for channel in range(light_curve.shape[0]):
                signal_data = light_curve[channel]
                median_val = np.median(signal_data)
                std_val = np.std(signal_data)
                
                # Points significantly below median might be transits
                transit_candidates = signal_data < (median_val - 2 * std_val)
                
                # Apply conservative augmentation only to non-transit regions
                non_transit_mask = ~transit_candidates
                
                # Add mild noise only to non-transit regions
                noise = np.random.normal(0, 0.005, signal_data.shape)
                augmented[channel, non_transit_mask] += noise[non_transit_mask]
        else:
            # Use provided transit mask
            non_transit_mask = ~self.transit_mask
            
            for channel in range(light_curve.shape[0]):
                # Add noise only to non-transit regions
                noise = np.random.normal(0, 0.005, light_curve.shape[-1])
                augmented[channel, non_transit_mask] += noise[non_transit_mask]
        
        metadata['transit_preserving_augmentation'] = True
        return augmented, metadata


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations."""
    
    def __init__(self, augmentations: List[BaseAugmentation], apply_probability: float = 0.8):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of augmentation techniques
            apply_probability: Probability of applying any augmentation
        """
        self.augmentations = augmentations
        self.apply_probability = apply_probability
    
    def __call__(self, light_curve: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Apply augmentation pipeline."""
        if metadata is None:
            metadata = {}
        
        # Decide whether to apply any augmentation
        if random.random() > self.apply_probability:
            return light_curve, metadata
        
        augmented = light_curve.copy()
        applied_augmentations = []
        
        # Apply each augmentation
        for aug in self.augmentations:
            augmented, metadata = aug(augmented, metadata)
            if any(key.startswith(aug.__class__.__name__.lower()) for key in metadata.keys()):
                applied_augmentations.append(aug.__class__.__name__)
        
        metadata['applied_augmentations'] = applied_augmentations
        return augmented, metadata


def create_standard_augmentation_pipeline(
    time_jitter_prob: float = 0.5,
    amplitude_scaling_prob: float = 0.5,
    gaussian_noise_prob: float = 0.6,
    random_masking_prob: float = 0.3,
    frequency_filtering_prob: float = 0.2,
    time_warping_prob: float = 0.2,
    mixup_prob: float = 0.3
) -> AugmentationPipeline:
    """
    Create standard augmentation pipeline for exoplanet detection.
    
    Args:
        time_jitter_prob: Probability of applying time jitter
        amplitude_scaling_prob: Probability of applying amplitude scaling
        gaussian_noise_prob: Probability of adding Gaussian noise
        random_masking_prob: Probability of applying random masking
        frequency_filtering_prob: Probability of applying frequency filtering
        time_warping_prob: Probability of applying time warping
        mixup_prob: Probability of applying MixUp
        
    Returns:
        Configured augmentation pipeline
    """
    augmentations = [
        TimeJitter(max_shift=50, probability=time_jitter_prob),
        AmplitudeScaling(scale_range=(0.85, 1.15), probability=amplitude_scaling_prob),
        GaussianNoise(noise_std=0.01, probability=gaussian_noise_prob),
        RandomMasking(mask_fraction=0.05, max_mask_length=20, probability=random_masking_prob),
        FrequencyFiltering(filter_type='lowpass', cutoff_range=(0.1, 0.4), probability=frequency_filtering_prob),
        TimeWarping(warp_strength=0.05, probability=time_warping_prob),
        MixUp(alpha=0.2, probability=mixup_prob)
    ]
    
    return AugmentationPipeline(augmentations, apply_probability=0.8)


def create_conservative_augmentation_pipeline() -> AugmentationPipeline:
    """
    Create conservative augmentation pipeline for sensitive applications.
    
    Returns:
        Conservative augmentation pipeline
    """
    augmentations = [
        TimeJitter(max_shift=20, probability=0.3),
        GaussianNoise(noise_std=0.005, probability=0.4),
        AmplitudeScaling(scale_range=(0.95, 1.05), probability=0.3),
        TransitPreservingAugmentation(probability=0.5)
    ]
    
    return AugmentationPipeline(augmentations, apply_probability=0.5)


def create_aggressive_augmentation_pipeline() -> AugmentationPipeline:
    """
    Create aggressive augmentation pipeline for data-scarce scenarios.
    
    Returns:
        Aggressive augmentation pipeline
    """
    augmentations = [
        TimeJitter(max_shift=100, probability=0.7),
        AmplitudeScaling(scale_range=(0.7, 1.3), probability=0.7),
        GaussianNoise(noise_std=0.02, probability=0.8),
        RandomMasking(mask_fraction=0.1, max_mask_length=50, probability=0.5),
        FrequencyFiltering(filter_type='lowpass', cutoff_range=(0.05, 0.5), probability=0.4),
        TimeWarping(warp_strength=0.15, probability=0.4),
        MixUp(alpha=0.4, probability=0.5)
    ]
    
    return AugmentationPipeline(augmentations, apply_probability=0.9)


def create_physics_aware_augmentation_pipeline() -> AugmentationPipeline:
    """
    Create physics-aware augmentation pipeline that preserves transit characteristics.
    
    Returns:
        Physics-aware augmentation pipeline
    """
    augmentations = [
        TransitPreservingAugmentation(probability=0.6),
        TimeJitter(max_shift=30, probability=0.4),
        GaussianNoise(noise_std=0.008, probability=0.5),
        AmplitudeScaling(scale_range=(0.9, 1.1), probability=0.4),
        FrequencyFiltering(filter_type='lowpass', cutoff_range=(0.15, 0.35), probability=0.3)
    ]
    
    return AugmentationPipeline(augmentations, apply_probability=0.7)