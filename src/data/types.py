"""
Core data types and structures for exoplanet detection pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch


@dataclass
class LightCurve:
    """Represents a single light curve observation from Kepler or TESS."""
    
    star_id: str
    time: np.ndarray          # Time stamps (days)
    flux: np.ndarray          # Normalized flux values
    flux_err: np.ndarray      # Flux uncertainties
    label: int               # 0=no planet, 1=planet
    period: Optional[float] = None   # Known period (if available)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate light curve data after initialization."""
        if len(self.time) != len(self.flux):
            raise ValueError("Time and flux arrays must have same length")
        
        if len(self.flux_err) != len(self.flux):
            raise ValueError("Flux error array must match flux length")
        
        if self.label not in [0, 1]:
            raise ValueError("Label must be 0 (no planet) or 1 (planet)")
        
        if len(self.flux) == 0:
            raise ValueError("Light curve cannot be empty")
    
    @property
    def length(self) -> int:
        """Return the number of data points in the light curve."""
        return len(self.flux)
    
    @property
    def duration(self) -> float:
        """Return the total duration of observations in days."""
        return float(np.max(self.time) - np.min(self.time))
    
    def get_snr(self) -> float:
        """Calculate signal-to-noise ratio of the light curve."""
        return float(np.median(self.flux) / np.median(self.flux_err))


@dataclass
class ProcessedLightCurve:
    """Preprocessed light curve ready for model input."""
    
    raw_flux: torch.Tensor      # Shape: (2048,)
    phase_folded_flux: torch.Tensor  # Shape: (2048,)
    mask: torch.Tensor          # Missing data mask
    label: int
    confidence_weight: float = 1.0    # For weighted loss
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate processed light curve data."""
        if self.raw_flux.shape[0] != 2048:
            raise ValueError("Raw flux must have length 2048")
        
        if self.phase_folded_flux.shape[0] != 2048:
            raise ValueError("Phase-folded flux must have length 2048")
        
        if self.mask.shape[0] != 2048:
            raise ValueError("Mask must have length 2048")
        
        if self.label not in [0, 1]:
            raise ValueError("Label must be 0 (no planet) or 1 (planet)")
        
        if self.confidence_weight <= 0:
            raise ValueError("Confidence weight must be positive")
    
    def to_dual_channel(self) -> torch.Tensor:
        """Convert to dual-channel tensor for model input."""
        return torch.stack([self.raw_flux, self.phase_folded_flux], dim=0)
    
    def apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply missing data mask to a tensor."""
        return tensor * self.mask


@dataclass
class TransitParams:
    """Transit parameters for synthetic generation using Mandel-Agol model."""
    
    period: float           # Orbital period (days)
    depth: float           # Transit depth (ppm)
    duration: float        # Transit duration (hours)
    impact_parameter: float # Impact parameter b
    limb_darkening: Tuple[float, float]  # u1, u2 coefficients
    epoch: float = 0.0     # Time of first transit (days)
    
    def __post_init__(self):
        """Validate transit parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        
        if self.depth <= 0:
            raise ValueError("Transit depth must be positive")
        
        if self.duration <= 0:
            raise ValueError("Transit duration must be positive")
        
        if not (0 <= self.impact_parameter <= 1.5):
            raise ValueError("Impact parameter must be between 0 and 1.5")
        
        u1, u2 = self.limb_darkening
        if not (0 <= u1 <= 1 and 0 <= u2 <= 1 and u1 + u2 <= 1):
            raise ValueError("Invalid limb-darkening coefficients")
    
    @classmethod
    def sample_realistic(cls, stellar_temp: float, stellar_radius: float) -> 'TransitParams':
        """Sample realistic transit parameters based on stellar properties."""
        # Realistic parameter ranges based on Kepler discoveries
        period = np.random.lognormal(np.log(10), 1.0)  # Log-normal around 10 days
        period = np.clip(period, 0.5, 500)  # Clip to reasonable range
        
        # Transit depth depends on planet/star radius ratio
        planet_radius_ratio = np.random.uniform(0.01, 0.2)  # Earth to Jupiter-sized
        depth = (planet_radius_ratio ** 2) * 1e6  # Convert to ppm
        
        # Duration scales with period and stellar properties
        duration = 0.2 * (period / 10) * (stellar_radius / 1.0)  # Hours
        duration = np.clip(duration, 0.5, 24)
        
        # Impact parameter - uniform distribution
        impact_parameter = np.random.uniform(0, 1.2)
        
        # Limb-darkening based on stellar temperature
        if stellar_temp > 6000:  # Hot stars
            u1, u2 = 0.3, 0.2
        elif stellar_temp > 5000:  # Sun-like stars
            u1, u2 = 0.4, 0.3
        else:  # Cool stars
            u1, u2 = 0.6, 0.1
        
        # Add some scatter
        u1 += np.random.normal(0, 0.1)
        u2 += np.random.normal(0, 0.1)
        u1, u2 = np.clip([u1, u2], 0, 1)
        
        # Ensure physical constraint
        if u1 + u2 > 1:
            u2 = 1 - u1
        
        return cls(
            period=period,
            depth=depth,
            duration=duration,
            impact_parameter=impact_parameter,
            limb_darkening=(u1, u2),
            epoch=np.random.uniform(0, period)
        )


@dataclass
class PredictionResult:
    """Model prediction with uncertainty and explanation."""
    
    star_id: str
    probability: float
    confidence_interval: Tuple[float, float]
    attribution_scores: np.ndarray
    explanation_text: str
    model_ensemble_votes: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate prediction result."""
        if not (0 <= self.probability <= 1):
            raise ValueError("Probability must be between 0 and 1")
        
        ci_low, ci_high = self.confidence_interval
        if not (0 <= ci_low <= ci_high <= 1):
            raise ValueError("Invalid confidence interval")
        
        if len(self.attribution_scores) == 0:
            raise ValueError("Attribution scores cannot be empty")
    
    @property
    def uncertainty(self) -> float:
        """Calculate prediction uncertainty from confidence interval."""
        ci_low, ci_high = self.confidence_interval
        return ci_high - ci_low
    
    @property
    def is_planet_candidate(self) -> bool:
        """Determine if this is a planet candidate (probability > 0.5)."""
        return self.probability > 0.5


# Configuration dataclass for preprocessing parameters
@dataclass
class PreprocessingConfig:
    """Configuration for light curve preprocessing."""
    
    target_length: int = 2048
    detrend_method: str = 'median'  # 'median' or 'savgol'
    normalization: str = 'zscore'   # 'zscore' or 'minmax'
    interpolation_method: str = 'linear'
    savgol_window: int = 101
    savgol_polyorder: int = 3
    median_filter_size: int = 49
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        if self.target_length <= 0:
            raise ValueError("Target length must be positive")
        
        if self.detrend_method not in ['median', 'savgol']:
            raise ValueError("Detrend method must be 'median' or 'savgol'")
        
        if self.normalization not in ['zscore', 'minmax']:
            raise ValueError("Normalization must be 'zscore' or 'minmax'")
        
        if self.savgol_window <= 0 or self.savgol_window % 2 == 0:
            raise ValueError("Savitzky-Golay window must be positive and odd")
        
        if self.savgol_polyorder >= self.savgol_window:
            raise ValueError("Polynomial order must be less than window size")


# Training configuration
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 5
    scheduler: str = 'cosine'  # 'cosine' or 'plateau'
    loss_function: str = 'bce'  # 'bce' or 'focal'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.patience <= 0:
            raise ValueError("Patience must be positive")
        
        if self.scheduler not in ['cosine', 'plateau']:
            raise ValueError("Scheduler must be 'cosine' or 'plateau'")
        
        if self.loss_function not in ['bce', 'focal']:
            raise ValueError("Loss function must be 'bce' or 'focal'")