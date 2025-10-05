"""
TESS data downloader with MAST API integration.
Provides functionality to download and process TESS sector data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import requests
import time
from tqdm import tqdm
import warnings

try:
    from astroquery.mast import Catalogs, Observations
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    warnings.warn("astroquery not available. TESS functionality will be limited.")

from .types import LightCurve
from .downloader import DataDownloader


class TESSDownloader(DataDownloader):
    """
    TESS data downloader with MAST API integration.
    
    Provides functionality to download TESS sector data with intelligent
    sector selection and data format standardization.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize TESS downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        super().__init__(cache_dir)
        self.base_url = "https://mast.stsci.edu/api/v0.1/"
        
        if not ASTROQUERY_AVAILABLE:
            raise ImportError(
                "astroquery is required for TESS data download. "
                "Install with: pip install astroquery"
            )
    
    def get_available_sectors(self) -> List[int]:
        """
        Get list of available TESS sectors.
        
        Returns:
            List of available sector numbers
        """
        try:
            # Query MAST for available TESS sectors
            obs_table = Observations.query_criteria(
                obs_collection="TESS",
                dataproduct_type="timeseries"
            )
            
            # Extract unique sectors
            sectors = []
            if len(obs_table) > 0 and 'sequence_number' in obs_table.colnames:
                sectors = sorted(list(set(obs_table['sequence_number'])))
            
            return sectors
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve TESS sectors: {e}")
            # Return default sectors if query fails
            return list(range(1, 70))  # TESS sectors 1-69 as of 2025
    
    def select_representative_sectors(self, n_sectors: int = 4) -> List[int]:
        """
        Select representative TESS sectors for analysis.
        
        Args:
            n_sectors: Number of sectors to select
            
        Returns:
            List of selected sector numbers
        """
        available_sectors = self.get_available_sectors()
        
        if len(available_sectors) <= n_sectors:
            return available_sectors
        
        # Select sectors distributed across the mission
        # Early mission: sectors 1-13 (southern hemisphere)
        # Later mission: sectors 14+ (northern hemisphere + extended)
        
        selected = []
        
        # Select from early mission (southern sky)
        early_sectors = [s for s in available_sectors if s <= 13]
        if early_sectors:
            selected.extend(np.linspace(
                min(early_sectors), max(early_sectors), 
                min(2, n_sectors//2), dtype=int
            ).tolist())
        
        # Select from later mission (northern sky + extended)
        later_sectors = [s for s in available_sectors if s > 13]
        if later_sectors:
            remaining = n_sectors - len(selected)
            if remaining > 0:
                selected.extend(np.linspace(
                    min(later_sectors), max(later_sectors),
                    min(remaining, len(later_sectors)), dtype=int
                ).tolist())
        
        # Remove duplicates and sort
        selected = sorted(list(set(selected)))
        
        # If we still need more sectors, add random ones
        while len(selected) < n_sectors and len(selected) < len(available_sectors):
            remaining = [s for s in available_sectors if s not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
        
        return selected[:n_sectors]
    
    def download_tess_sectors(
        self,
        sectors: List[int],
        output_dir: Path,
        sample_size: Optional[int] = None,
        force_download: bool = False
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Download TESS data for specified sectors.
        
        Args:
            sectors: List of sector numbers to download
            output_dir: Directory to save downloaded data
            sample_size: Maximum number of targets per sector
            force_download: Whether to re-download existing data
            
        Returns:
            Tuple of (metadata DataFrame, metadata file path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_dir / 'tess_metadata.csv'
        
        # Check if data already exists
        if metadata_file.exists() and not force_download:
            self.logger.info(f"Loading existing TESS metadata from {metadata_file}")
            return pd.read_csv(metadata_file), metadata_file
        
        all_metadata = []
        
        for sector in tqdm(sectors, desc="Downloading TESS sectors"):
            try:
                sector_metadata = self._download_sector_data(
                    sector, output_dir, sample_size
                )
                all_metadata.extend(sector_metadata)
                
                # Add delay to avoid overwhelming the server
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to download sector {sector}: {e}")
                continue
        
        # Create combined metadata DataFrame
        if all_metadata:
            metadata_df = pd.DataFrame(all_metadata)
            
            # Standardize column names
            metadata_df = self._standardize_tess_metadata(metadata_df)
            
            # Save metadata
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved TESS metadata to {metadata_file}")
            
            return metadata_df, metadata_file
        
        else:
            # Return empty DataFrame if no data downloaded
            empty_df = pd.DataFrame(columns=[
                'tic_id', 'sector', 'ra', 'dec', 'tmag', 'teff', 'radius', 'mass',
                'disposition', 'period', 'epoch', 'depth', 'duration', 'label'
            ])
            return empty_df, metadata_file
    
    def _download_sector_data(
        self, 
        sector: int, 
        output_dir: Path, 
        sample_size: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Download data for a single TESS sector.
        
        Args:
            sector: Sector number
            output_dir: Output directory
            sample_size: Maximum number of targets
            
        Returns:
            List of metadata dictionaries
        """
        try:
            # Query TESS Input Catalog (TIC) for targets in this sector
            obs_table = Observations.query_criteria(
                obs_collection="TESS",
                sequence_number=sector,
                dataproduct_type="timeseries"
            )
            
            if len(obs_table) == 0:
                self.logger.warning(f"No observations found for sector {sector}")
                return []
            
            # Limit sample size if specified
            if sample_size and len(obs_table) > sample_size:
                # Random sampling for diversity
                indices = np.random.choice(
                    len(obs_table), size=sample_size, replace=False
                )
                obs_table = obs_table[indices]
            
            metadata_list = []
            
            for obs in tqdm(obs_table, desc=f"Processing sector {sector}", leave=False):
                try:
                    # Extract TIC ID
                    tic_id = self._extract_tic_id(obs)
                    if tic_id is None:
                        continue
                    
                    # Get stellar parameters from TIC
                    stellar_params = self._get_tic_parameters(tic_id)
                    
                    # Check for known planets
                    planet_info = self._check_for_planets(tic_id)
                    
                    # Create metadata entry
                    metadata = {
                        'tic_id': tic_id,
                        'sector': sector,
                        'ra': obs.get('s_ra', np.nan),
                        'dec': obs.get('s_dec', np.nan),
                        'tmag': stellar_params.get('Tmag', np.nan),
                        'teff': stellar_params.get('Teff', np.nan),
                        'radius': stellar_params.get('rad', np.nan),
                        'mass': stellar_params.get('mass', np.nan),
                        'disposition': planet_info.get('disposition', 'UNKNOWN'),
                        'period': planet_info.get('period', np.nan),
                        'epoch': planet_info.get('epoch', np.nan),
                        'depth': planet_info.get('depth', np.nan),
                        'duration': planet_info.get('duration', np.nan),
                        'label': 1 if planet_info.get('disposition') == 'CONFIRMED' else 0
                    }
                    
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing TIC {tic_id}: {e}")
                    continue
            
            return metadata_list
            
        except Exception as e:
            self.logger.error(f"Error downloading sector {sector}: {e}")
            return []
    
    def _extract_tic_id(self, obs_row) -> Optional[int]:
        """Extract TIC ID from observation row."""
        try:
            # Try different possible column names
            for col in ['target_name', 'obs_id', 'obsid']:
                if col in obs_row.colnames:
                    target_name = str(obs_row[col])
                    # Extract TIC ID from target name
                    if 'TIC' in target_name:
                        tic_str = target_name.split('TIC')[-1].strip()
                        # Remove any non-numeric characters
                        tic_str = ''.join(filter(str.isdigit, tic_str))
                        if tic_str:
                            return int(tic_str)
            return None
        except:
            return None
    
    def _get_tic_parameters(self, tic_id: int) -> Dict[str, float]:
        """
        Get stellar parameters from TIC catalog.
        
        Args:
            tic_id: TIC identifier
            
        Returns:
            Dictionary of stellar parameters
        """
        try:
            # Query TIC catalog
            tic_data = Catalogs.query_criteria(
                catalog="Tic",
                ID=tic_id
            )
            
            if len(tic_data) > 0:
                row = tic_data[0]
                return {
                    'Tmag': row.get('Tmag', np.nan),
                    'Teff': row.get('Teff', np.nan),
                    'rad': row.get('rad', np.nan),
                    'mass': row.get('mass', np.nan),
                    'logg': row.get('logg', np.nan),
                    'MH': row.get('MH', np.nan)
                }
            
        except Exception as e:
            self.logger.debug(f"Could not get TIC parameters for {tic_id}: {e}")
        
        return {}
    
    def _check_for_planets(self, tic_id: int) -> Dict[str, Any]:
        """
        Check if TIC target has known planets.
        
        Args:
            tic_id: TIC identifier
            
        Returns:
            Dictionary with planet information
        """
        try:
            # Query for TESS Objects of Interest (TOI)
            toi_data = Catalogs.query_criteria(
                catalog="Toi",
                TIC=tic_id
            )
            
            if len(toi_data) > 0:
                # Use the first (most significant) planet candidate
                row = toi_data[0]
                return {
                    'disposition': row.get('TFOPWG Disposition', 'CANDIDATE'),
                    'period': row.get('Period (days)', np.nan),
                    'epoch': row.get('Epoch (BJD)', np.nan),
                    'depth': row.get('Depth (ppm)', np.nan),
                    'duration': row.get('Duration (hours)', np.nan)
                }
            
        except Exception as e:
            self.logger.debug(f"Could not check planets for TIC {tic_id}: {e}")
        
        return {'disposition': 'UNKNOWN'}
    
    def _standardize_tess_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize TESS metadata to match Kepler format.
        
        Args:
            df: Raw TESS metadata DataFrame
            
        Returns:
            Standardized metadata DataFrame
        """
        # Create standardized columns
        standardized = df.copy()
        
        # Map TESS columns to standard names
        column_mapping = {
            'tic_id': 'star_id',
            'tmag': 'magnitude',
            'teff': 'temperature',
            'disposition': 'koi_disposition'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in standardized.columns:
                standardized[new_col] = standardized[old_col]
        
        # Ensure required columns exist
        required_columns = ['star_id', 'label', 'magnitude', 'temperature']
        for col in required_columns:
            if col not in standardized.columns:
                standardized[col] = np.nan
        
        # Convert dispositions to binary labels
        if 'koi_disposition' in standardized.columns:
            standardized['label'] = (
                standardized['koi_disposition'].str.contains('CONFIRMED', na=False)
            ).astype(int)
        
        return standardized
    
    def create_mock_tess_data(
        self,
        n_samples: int = 1000,
        output_dir: Path = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Create mock TESS data for testing and demonstration.
        
        Args:
            n_samples: Number of mock samples to create
            output_dir: Output directory for mock data
            
        Returns:
            Tuple of (metadata DataFrame, metadata file path)
        """
        if output_dir is None:
            output_dir = self.cache_dir / 'mock_tess'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate mock TESS metadata
        np.random.seed(42)
        
        sectors = self.select_representative_sectors(4)
        
        mock_data = []
        for i in range(n_samples):
            # Random TIC ID
            tic_id = np.random.randint(100000000, 999999999)
            
            # Random sector
            sector = np.random.choice(sectors)
            
            # Random coordinates
            ra = np.random.uniform(0, 360)
            dec = np.random.uniform(-90, 90)
            
            # Stellar parameters (realistic distributions)
            tmag = np.random.normal(12, 2)
            teff = np.random.normal(5500, 1000)
            radius = np.random.lognormal(0, 0.3)
            mass = np.random.lognormal(0, 0.2)
            
            # Planet parameters (15% have planets)
            has_planet = np.random.random() < 0.15
            
            if has_planet:
                disposition = 'CONFIRMED'
                period = np.random.lognormal(np.log(10), 1)
                depth = np.random.uniform(100, 10000)  # ppm
                duration = np.random.uniform(1, 12)  # hours
                label = 1
            else:
                disposition = 'FALSE POSITIVE'
                period = np.nan
                depth = np.nan
                duration = np.nan
                label = 0
            
            mock_data.append({
                'tic_id': tic_id,
                'sector': sector,
                'ra': ra,
                'dec': dec,
                'tmag': tmag,
                'teff': teff,
                'radius': radius,
                'mass': mass,
                'disposition': disposition,
                'period': period,
                'depth': depth,
                'duration': duration,
                'label': label,
                'star_id': f'TIC_{tic_id}',
                'magnitude': tmag,
                'temperature': teff,
                'koi_disposition': disposition
            })
        
        # Create DataFrame
        df = pd.DataFrame(mock_data)
        
        # Save to file
        metadata_file = output_dir / 'tess_mock_metadata.csv'
        df.to_csv(metadata_file, index=False)
        
        self.logger.info(f"Created mock TESS data: {len(df)} samples")
        self.logger.info(f"Saved to {metadata_file}")
        
        return df, metadata_file


def create_tess_downloader(cache_dir: Optional[Path] = None) -> TESSDownloader:
    """
    Factory function to create TESS downloader.
    
    Args:
        cache_dir: Directory for caching data
        
    Returns:
        Configured TESS downloader
    """
    return TESSDownloader(cache_dir=cache_dir)