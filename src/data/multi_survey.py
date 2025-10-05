"""
Multi-survey integration framework for exoplanet detection.
Supports Kepler, TESS, K2, PLATO, and ground-based surveys.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from .types import LightCurve, PreprocessingConfig
from .downloader import DataDownloader
from .tess_downloader import TESSDownloader
from ..preprocessing.preprocessor import LightCurvePreprocessor


@dataclass
class SurveyMetadata:
    """Metadata for astronomical surveys."""
    
    name: str
    cadence: float  # seconds
    duration: float  # days
    magnitude_range: Tuple[float, float]
    wavelength_band: str
    precision: float  # ppm
    sky_coverage: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'cadence': self.cadence,
            'duration': self.duration,
            'magnitude_range': self.magnitude_range,
            'wavelength_band': self.wavelength_band,
            'precision': self.precision,
            'sky_coverage': self.sky_coverage
        }


class SurveyAdapter(ABC):
    """Abstract base class for survey data adapters."""
    
    @abstractmethod
    def get_metadata(self) -> SurveyMetadata:
        """Get survey metadata."""
        pass
    
    @abstractmethod
    def download_data(
        self, 
        output_dir: Path, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """Download survey data."""
        pass
    
    @abstractmethod
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format."""
        pass
    
    @abstractmethod
    def create_light_curve(self, row: pd.Series) -> Optional[LightCurve]:
        """Create light curve from data row."""
        pass


class KeplerAdapter(SurveyAdapter):
    """Adapter for Kepler survey data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Kepler adapter."""
        self.downloader = DataDownloader(cache_dir=cache_dir)
        self.logger = logging.getLogger(__name__)
    
    def get_metadata(self) -> SurveyMetadata:
        """Get Kepler survey metadata."""
        return SurveyMetadata(
            name="Kepler",
            cadence=1800,  # 30 minutes
            duration=1460,  # ~4 years
            magnitude_range=(9.0, 16.0),
            wavelength_band="Kepler (420-900 nm)",
            precision=20,  # ppm
            sky_coverage="105 square degrees"
        )
    
    def download_data(
        self, 
        output_dir: Path, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """Download Kepler KOI data."""
        return self.downloader.download_kepler_koi(
            output_dir=output_dir,
            sample_size=sample_size
        )
    
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize Kepler data format."""
        standardized = raw_data.copy()
        
        # Ensure required columns
        column_mapping = {
            'kepid': 'star_id',
            'koi_disposition': 'disposition',
            'koi_kepmag': 'magnitude',
            'koi_teff': 'temperature',
            'koi_period': 'period',
            'koi_depth': 'depth',
            'koi_duration': 'duration'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in standardized.columns:
                standardized[new_col] = standardized[old_col]
        
        # Add survey identifier
        standardized['survey'] = 'Kepler'
        
        return standardized
    
    def create_light_curve(self, row: pd.Series) -> Optional[LightCurve]:
        """Create light curve from Kepler data row."""
        try:
            # Generate mock light curve data
            # In practice, this would load actual Kepler light curve files
            time = np.linspace(0, 90, 2048)  # 90 days
            flux = np.random.normal(1.0, 0.01, 2048)
            flux_err = np.full_like(flux, 0.01)
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                star_id=str(row.get('star_id', 'unknown'))
            )
        except Exception as e:
            self.logger.warning(f"Failed to create light curve: {e}")
            return None


class TESSAdapter(SurveyAdapter):
    """Adapter for TESS survey data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize TESS adapter."""
        self.downloader = TESSDownloader(cache_dir=cache_dir)
        self.logger = logging.getLogger(__name__)
    
    def get_metadata(self) -> SurveyMetadata:
        """Get TESS survey metadata."""
        return SurveyMetadata(
            name="TESS",
            cadence=1800,  # 30 minutes (2-minute for some targets)
            duration=27.4,  # per sector
            magnitude_range=(4.0, 16.0),
            wavelength_band="TESS (600-1000 nm)",
            precision=60,  # ppm for 10th magnitude
            sky_coverage="Full sky (13 sectors per hemisphere)"
        )
    
    def download_data(
        self, 
        output_dir: Path, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """Download TESS sector data."""
        sectors = self.downloader.select_representative_sectors(4)
        return self.downloader.download_tess_sectors(
            sectors=sectors,
            output_dir=output_dir,
            sample_size=sample_size
        )
    
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize TESS data format."""
        standardized = raw_data.copy()
        
        # TESS-specific standardization
        if 'tic_id' in standardized.columns:
            standardized['star_id'] = 'TIC_' + standardized['tic_id'].astype(str)
        
        # Add survey identifier
        standardized['survey'] = 'TESS'
        
        return standardized
    
    def create_light_curve(self, row: pd.Series) -> Optional[LightCurve]:
        """Create light curve from TESS data row."""
        try:
            # Generate mock TESS light curve
            time = np.linspace(0, 27, 1296)  # 27 days, 30-min cadence
            flux = np.random.normal(1.0, 0.005, 1296)
            flux_err = np.full_like(flux, 0.005)
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                star_id=str(row.get('star_id', 'unknown'))
            )
        except Exception as e:
            self.logger.warning(f"Failed to create TESS light curve: {e}")
            return None


class K2Adapter(SurveyAdapter):
    """Adapter for K2 survey data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize K2 adapter."""
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
    
    def get_metadata(self) -> SurveyMetadata:
        """Get K2 survey metadata."""
        return SurveyMetadata(
            name="K2",
            cadence=1800,  # 30 minutes
            duration=80,  # per campaign
            magnitude_range=(8.0, 18.0),
            wavelength_band="Kepler (420-900 nm)",
            precision=50,  # ppm
            sky_coverage="Various fields along ecliptic"
        )
    
    def download_data(
        self, 
        output_dir: Path, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """Download K2 data (mock implementation)."""
        # Mock K2 data
        n_samples = sample_size or 500
        
        mock_data = []
        for i in range(n_samples):
            mock_data.append({
                'epic_id': 200000000 + i,
                'campaign': np.random.randint(0, 20),
                'k2_kepmag': np.random.normal(12, 2),
                'k2_teff': np.random.normal(5500, 1000),
                'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']),
                'period': np.random.lognormal(np.log(10), 1) if np.random.random() < 0.1 else np.nan
            })
        
        df = pd.DataFrame(mock_data)
        metadata_file = output_dir / 'k2_metadata.csv'
        df.to_csv(metadata_file, index=False)
        
        return df, metadata_file
    
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize K2 data format."""
        standardized = raw_data.copy()
        
        # K2-specific mapping
        column_mapping = {
            'epic_id': 'star_id',
            'k2_kepmag': 'magnitude',
            'k2_teff': 'temperature'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in standardized.columns:
                standardized[new_col] = standardized[old_col]
        
        standardized['survey'] = 'K2'
        return standardized
    
    def create_light_curve(self, row: pd.Series) -> Optional[LightCurve]:
        """Create light curve from K2 data row."""
        try:
            time = np.linspace(0, 80, 3840)  # 80 days
            flux = np.random.normal(1.0, 0.02, 3840)
            flux_err = np.full_like(flux, 0.02)
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                star_id=str(row.get('star_id', 'unknown'))
            )
        except Exception as e:
            self.logger.warning(f"Failed to create K2 light curve: {e}")
            return None


class PLATOAdapter(SurveyAdapter):
    """Adapter for PLATO survey data (future mission)."""
    
    def get_metadata(self) -> SurveyMetadata:
        """Get PLATO survey metadata."""
        return SurveyMetadata(
            name="PLATO",
            cadence=25,  # 25 seconds
            duration=1460,  # 4 years
            magnitude_range=(4.0, 16.0),
            wavelength_band="PLATO (500-1000 nm)",
            precision=27,  # ppm for Sun-like stars
            sky_coverage="2232 square degrees"
        )
    
    def download_data(
        self, 
        output_dir: Path, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """Download PLATO data (mock - future mission)."""
        # Mock PLATO data for future compatibility
        n_samples = sample_size or 1000
        
        mock_data = []
        for i in range(n_samples):
            mock_data.append({
                'plato_id': f'PLATO_{i:08d}',
                'magnitude': np.random.normal(10, 3),
                'temperature': np.random.normal(5778, 800),
                'disposition': 'SIMULATED',
                'period': np.random.lognormal(np.log(20), 1) if np.random.random() < 0.2 else np.nan
            })
        
        df = pd.DataFrame(mock_data)
        metadata_file = output_dir / 'plato_metadata.csv'
        df.to_csv(metadata_file, index=False)
        
        return df, metadata_file
    
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize PLATO data format."""
        standardized = raw_data.copy()
        standardized['survey'] = 'PLATO'
        
        if 'plato_id' in standardized.columns:
            standardized['star_id'] = standardized['plato_id']
        
        return standardized
    
    def create_light_curve(self, row: pd.Series) -> Optional[LightCurve]:
        """Create light curve from PLATO data row."""
        try:
            # High-cadence PLATO data
            time = np.linspace(0, 365, 1262304)  # 1 year, 25-second cadence
            flux = np.random.normal(1.0, 0.0001, 1262304)  # Very high precision
            flux_err = np.full_like(flux, 0.0001)
            
            return LightCurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                star_id=str(row.get('star_id', 'unknown'))
            )
        except Exception as e:
            self.logger.warning(f"Failed to create PLATO light curve: {e}")
            return None


class MultiSurveyManager:
    """
    Manager for multi-survey data integration and processing.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize multi-survey manager.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize survey adapters
        self.adapters = {
            'kepler': KeplerAdapter(cache_dir),
            'tess': TESSAdapter(cache_dir),
            'k2': K2Adapter(cache_dir),
            'plato': PLATOAdapter()
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_available_surveys(self) -> List[str]:
        """Get list of available surveys."""
        return list(self.adapters.keys())
    
    def get_survey_metadata(self, survey: str) -> SurveyMetadata:
        """
        Get metadata for specific survey.
        
        Args:
            survey: Survey name
            
        Returns:
            Survey metadata
        """
        if survey not in self.adapters:
            raise ValueError(f"Unknown survey: {survey}")
        
        return self.adapters[survey].get_metadata()
    
    def download_multi_survey_data(
        self,
        surveys: List[str],
        output_dir: Path,
        sample_size_per_survey: Optional[int] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Path]]:
        """
        Download data from multiple surveys.
        
        Args:
            surveys: List of survey names
            output_dir: Output directory
            sample_size_per_survey: Sample size per survey
            
        Returns:
            Dictionary mapping survey names to (dataframe, file_path) tuples
        """
        results = {}
        
        for survey in surveys:
            if survey not in self.adapters:
                self.logger.warning(f"Unknown survey: {survey}")
                continue
            
            try:
                survey_dir = output_dir / survey
                survey_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Downloading {survey} data...")
                
                df, file_path = self.adapters[survey].download_data(
                    survey_dir, sample_size_per_survey
                )
                
                # Standardize format
                df_standardized = self.adapters[survey].standardize_format(df)
                
                results[survey] = (df_standardized, file_path)
                
                self.logger.info(f"Downloaded {len(df)} samples from {survey}")
                
            except Exception as e:
                self.logger.error(f"Failed to download {survey} data: {e}")
                continue
        
        return results
    
    def create_unified_dataset(
        self,
        survey_data: Dict[str, Tuple[pd.DataFrame, Path]],
        output_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create unified dataset from multiple surveys.
        
        Args:
            survey_data: Dictionary of survey data
            output_file: Optional output file path
            
        Returns:
            Unified dataset
        """
        unified_dfs = []
        
        for survey, (df, _) in survey_data.items():
            # Ensure consistent columns
            df_copy = df.copy()
            
            # Add survey-specific metadata
            df_copy['survey'] = survey
            df_copy['survey_metadata'] = df_copy.apply(
                lambda row: self.get_survey_metadata(survey).to_dict(), axis=1
            )
            
            unified_dfs.append(df_copy)
        
        # Combine all datasets
        unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
        
        # Save if output file specified
        if output_file:
            unified_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved unified dataset to {output_file}")
        
        return unified_df
    
    def create_cross_survey_preprocessor(
        self,
        target_cadence: float = 1800,  # 30 minutes
        target_length: int = 2048
    ) -> LightCurvePreprocessor:
        """
        Create preprocessor for cross-survey compatibility.
        
        Args:
            target_cadence: Target cadence in seconds
            target_length: Target sequence length
            
        Returns:
            Configured preprocessor
        """
        config = PreprocessingConfig(
            target_length=target_length,
            detrend_method='median',
            normalization='zscore',
            handle_gaps=True,
            resample_cadence=target_cadence
        )
        
        return LightCurvePreprocessor(config)
    
    def get_survey_statistics(
        self, 
        unified_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each survey in unified dataset.
        
        Args:
            unified_df: Unified dataset
            
        Returns:
            Survey statistics
        """
        stats = {}
        
        for survey in unified_df['survey'].unique():
            survey_df = unified_df[unified_df['survey'] == survey]
            
            stats[survey] = {
                'sample_count': len(survey_df),
                'planet_fraction': survey_df['label'].mean() if 'label' in survey_df.columns else 0,
                'magnitude_range': (
                    survey_df['magnitude'].min(), 
                    survey_df['magnitude'].max()
                ) if 'magnitude' in survey_df.columns else (np.nan, np.nan),
                'temperature_range': (
                    survey_df['temperature'].min(),
                    survey_df['temperature'].max()
                ) if 'temperature' in survey_df.columns else (np.nan, np.nan)
            }
        
        return stats


# Factory functions
def create_multi_survey_manager(cache_dir: Optional[Path] = None) -> MultiSurveyManager:
    """Create multi-survey manager."""
    return MultiSurveyManager(cache_dir=cache_dir)


def get_survey_adapter(survey: str, cache_dir: Optional[Path] = None) -> SurveyAdapter:
    """Get specific survey adapter."""
    adapters = {
        'kepler': KeplerAdapter,
        'tess': TESSAdapter,
        'k2': K2Adapter,
        'plato': PLATOAdapter
    }
    
    if survey not in adapters:
        raise ValueError(f"Unknown survey: {survey}")
    
    return adapters[survey](cache_dir)