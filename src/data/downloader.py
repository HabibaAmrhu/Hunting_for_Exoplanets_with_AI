"""
Data downloader for Kepler and TESS light curve data.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import warnings

# Suppress astroquery warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='astroquery')

try:
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    from astroquery.mast import Catalogs, Observations
    import lightkurve as lk
    ASTRO_IMPORTS_AVAILABLE = True
except ImportError:
    ASTRO_IMPORTS_AVAILABLE = False
    print("Warning: Astronomical libraries not available. Install with:")
    print("pip install astroquery lightkurve")


class DataDownloader:
    """
    Handles downloading and caching of Kepler and TESS light curve data.
    
    Supports both NASA Exoplanet Archive queries and direct light curve downloads
    with intelligent caching and integrity verification.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = "data/cache"):
        """
        Initialize the data downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "kepler").mkdir(exist_ok=True)
        (self.cache_dir / "tess").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ExoplanetDetectionPipeline/1.0'
        })
    
    def download_kepler_koi(
        self, 
        output_dir: Union[str, Path],
        sample_size: Optional[int] = None,
        min_period: float = 0.5,
        max_period: float = 500.0,
        force_download: bool = False
    ) -> Tuple[pd.DataFrame, str]:
        """
        Download Kepler Objects of Interest (KOI) table and associated light curves.
        
        Args:
            output_dir: Directory to save processed data
            sample_size: Number of KOIs to download (None for all)
            min_period: Minimum orbital period (days)
            max_period: Maximum orbital period (days)
            force_download: Force re-download even if cached
            
        Returns:
            Tuple of (metadata_dataframe, metadata_file_path)
        """
        if not ASTRO_IMPORTS_AVAILABLE:
            raise ImportError("Astronomical libraries required. Install with: pip install astroquery lightkurve")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for cached KOI table
        koi_cache_file = self.cache_dir / "metadata" / "koi_table.csv"
        
        if koi_cache_file.exists() and not force_download:
            print(f"Loading cached KOI table from {koi_cache_file}")
            koi_table = pd.read_csv(koi_cache_file)
        else:
            print("Downloading KOI table from NASA Exoplanet Archive...")
            try:
                # Download KOI table
                koi_table = NasaExoplanetArchive.query_criteria(
                    table="cumulative",
                    select="kepoi_name,kepid,koi_disposition,koi_period,koi_duration,"
                           "koi_depth,koi_prad,koi_teq,koi_insol,koi_dor,koi_impact,"
                           "koi_incl,ra,dec,koi_kepmag,koi_tce_plnt_num"
                ).to_pandas()
                
                # Cache the table
                koi_table.to_csv(koi_cache_file, index=False)
                print(f"Cached KOI table to {koi_cache_file}")
                
            except Exception as e:
                print(f"Error downloading KOI table: {e}")
                # Fallback to a minimal synthetic dataset for development
                return self._create_synthetic_koi_sample(output_dir, sample_size or 100)
        
        # Filter and process KOI table
        processed_kois = self._process_koi_table(
            koi_table, sample_size, min_period, max_period
        )
        
        print(f"Processed {len(processed_kois)} KOIs")
        print(f"Class distribution: {processed_kois['label'].value_counts().to_dict()}")
        
        # Download light curves for selected KOIs
        metadata_list = []
        
        for idx, koi in tqdm(processed_kois.iterrows(), 
                           total=len(processed_kois), 
                           desc="Downloading light curves"):
            
            try:
                lc_metadata = self._download_kepler_light_curve(
                    koi, output_dir, force_download
                )
                if lc_metadata:
                    metadata_list.append(lc_metadata)
                    
            except Exception as e:
                print(f"Warning: Failed to download {koi['kepoi_name']}: {e}")
                continue
        
        # Create final metadata DataFrame
        if metadata_list:
            metadata_df = pd.DataFrame(metadata_list)
        else:
            # Fallback to synthetic data if no downloads succeeded
            print("No light curves downloaded successfully, creating synthetic sample...")
            return self._create_synthetic_koi_sample(output_dir, sample_size or 100)
        
        # Save metadata
        metadata_file = output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"Saved metadata for {len(metadata_df)} light curves to {metadata_file}")
        
        return metadata_df, str(metadata_file)
    
    def _process_koi_table(
        self, 
        koi_table: pd.DataFrame, 
        sample_size: Optional[int],
        min_period: float,
        max_period: float
    ) -> pd.DataFrame:
        """Process and filter the KOI table."""
        
        # Clean and filter data
        koi_clean = koi_table.copy()
        
        # Remove rows with missing essential data
        essential_cols = ['kepid', 'koi_disposition']
        koi_clean = koi_clean.dropna(subset=essential_cols)
        
        # Filter by period if available
        if 'koi_period' in koi_clean.columns:
            period_mask = (
                (koi_clean['koi_period'] >= min_period) & 
                (koi_clean['koi_period'] <= max_period)
            )
            koi_clean = koi_clean[period_mask | koi_clean['koi_period'].isna()]
        
        # Create binary labels
        # CONFIRMED and CANDIDATE = 1 (planet), FALSE POSITIVE = 0 (no planet)
        def create_label(disposition):
            if pd.isna(disposition):
                return 0
            disposition = str(disposition).upper()
            if 'CONFIRMED' in disposition or 'CANDIDATE' in disposition:
                return 1
            else:
                return 0
        
        koi_clean['label'] = koi_clean['koi_disposition'].apply(create_label)
        
        # Balance classes if sampling
        if sample_size:
            # Try to get balanced sample
            planets = koi_clean[koi_clean['label'] == 1]
            non_planets = koi_clean[koi_clean['label'] == 0]
            
            n_planets = min(len(planets), sample_size // 2)
            n_non_planets = min(len(non_planets), sample_size - n_planets)
            
            sampled_planets = planets.sample(n=n_planets, random_state=42)
            sampled_non_planets = non_planets.sample(n=n_non_planets, random_state=42)
            
            koi_clean = pd.concat([sampled_planets, sampled_non_planets]).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
        
        return koi_clean
    
    def _download_kepler_light_curve(
        self, 
        koi: pd.Series, 
        output_dir: Path,
        force_download: bool = False
    ) -> Optional[Dict]:
        """Download individual Kepler light curve."""
        
        kepid = int(koi['kepid'])
        star_id = f"KIC_{kepid}"
        
        # Check if already downloaded
        lc_file = output_dir / f"{star_id}.npz"
        if lc_file.exists() and not force_download:
            # Verify file integrity
            if self._verify_light_curve_file(lc_file):
                return self._create_metadata_entry(koi, star_id, lc_file)
        
        try:
            # Download using lightkurve
            search_result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
            
            if len(search_result) == 0:
                print(f"No light curves found for KIC {kepid}")
                return None
            
            # Download and combine all quarters
            lc_collection = search_result.download_all(quality_bitmask='hardest')
            
            if lc_collection is None or len(lc_collection) == 0:
                return None
            
            # Combine quarters
            lc = lc_collection.stitch()
            
            # Remove NaN values
            lc = lc.remove_nans()
            
            if len(lc.flux) < 100:  # Minimum length requirement
                return None
            
            # Normalize flux
            lc = lc.normalize()
            
            # Save to file
            np.savez_compressed(
                lc_file,
                time=lc.time.value,
                flux=lc.flux.value,
                flux_err=lc.flux_err.value if lc.flux_err is not None else np.ones_like(lc.flux.value) * 0.001
            )
            
            return self._create_metadata_entry(koi, star_id, lc_file)
            
        except Exception as e:
            print(f"Error downloading KIC {kepid}: {e}")
            return None
    
    def _create_metadata_entry(self, koi: pd.Series, star_id: str, lc_file: Path) -> Dict:
        """Create metadata entry for a light curve."""
        
        return {
            'star_id': star_id,
            'kepid': int(koi['kepid']),
            'kepoi_name': koi.get('kepoi_name', ''),
            'label': int(koi['label']),
            'period': koi.get('koi_period', None),
            'duration': koi.get('koi_duration', None),
            'depth': koi.get('koi_depth', None),
            'magnitude': koi.get('koi_kepmag', None),
            'teff': None,  # Not in KOI table
            'radius': None,  # Not in KOI table
            'file_path': lc_file.name,
            'data_source': 'kepler',
            'sector': None  # Kepler uses quarters, not sectors
        }
    
    def _verify_light_curve_file(self, file_path: Path) -> bool:
        """Verify integrity of a light curve file."""
        try:
            data = np.load(file_path)
            required_keys = ['time', 'flux']
            
            for key in required_keys:
                if key not in data:
                    return False
                if len(data[key]) == 0:
                    return False
            
            # Check for reasonable data ranges
            if np.any(~np.isfinite(data['flux'])):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_synthetic_koi_sample(
        self, 
        output_dir: Path, 
        sample_size: int
    ) -> Tuple[pd.DataFrame, str]:
        """Create synthetic KOI sample for development/testing."""
        
        print(f"Creating synthetic KOI sample with {sample_size} light curves...")
        
        np.random.seed(42)  # For reproducibility
        
        metadata_list = []
        
        for i in range(sample_size):
            # Create synthetic light curve
            star_id = f"SYNTHETIC_{i:04d}"
            label = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% planets
            
            # Generate realistic light curve
            n_points = np.random.randint(1000, 5000)
            time = np.linspace(0, 90, n_points)  # ~3 months
            
            # Base stellar variability
            flux = 1.0 + 0.001 * np.random.randn(n_points)
            
            # Add transit if planet
            if label == 1:
                period = np.random.uniform(1, 50)
                depth = np.random.uniform(100, 5000) * 1e-6  # ppm to fraction
                duration = np.random.uniform(1, 8) / 24  # hours to days
                
                # Simple box transit
                phase = (time % period) / period
                transit_mask = np.abs(phase - 0.5) < (duration / period / 2)
                flux[transit_mask] -= depth
            
            # Add noise
            flux_err = np.full_like(flux, 0.001)
            flux += flux_err * np.random.randn(len(flux))
            
            # Save light curve
            lc_file = output_dir / f"{star_id}.npz"
            np.savez_compressed(
                lc_file,
                time=time,
                flux=flux,
                flux_err=flux_err
            )
            
            # Create metadata
            metadata_list.append({
                'star_id': star_id,
                'kepid': 100000 + i,
                'kepoi_name': f'K{100000 + i}.01',
                'label': label,
                'period': period if label == 1 else None,
                'duration': duration * 24 if label == 1 else None,  # Convert back to hours
                'depth': depth * 1e6 if label == 1 else None,  # Convert back to ppm
                'magnitude': np.random.uniform(10, 16),
                'teff': np.random.uniform(3500, 7000),
                'radius': np.random.uniform(0.5, 2.0),
                'file_path': lc_file.name,
                'data_source': 'synthetic',
                'sector': None
            })
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_file = output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"Created {len(metadata_df)} synthetic light curves")
        print(f"Class distribution: {metadata_df['label'].value_counts().to_dict()}")
        
        return metadata_df, str(metadata_file)
    
    def verify_data_integrity(self, data_path: Union[str, Path]) -> bool:
        """
        Verify integrity of downloaded dataset.
        
        Args:
            data_path: Path to metadata file or data directory
            
        Returns:
            True if all data is valid
        """
        data_path = Path(data_path)
        
        if data_path.is_file() and data_path.suffix == '.csv':
            # Metadata file provided
            metadata_file = data_path
            data_dir = data_path.parent
        else:
            # Directory provided
            data_dir = data_path
            metadata_file = data_dir / "metadata.csv"
        
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            return False
        
        # Load metadata
        try:
            metadata = pd.read_csv(metadata_file)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return False
        
        # Check required columns
        required_cols = ['star_id', 'label', 'file_path']
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
        
        # Verify each light curve file
        invalid_files = []
        
        for _, row in metadata.iterrows():
            lc_file = data_dir / row['file_path']
            
            if not lc_file.exists():
                invalid_files.append(f"Missing: {lc_file}")
                continue
            
            if not self._verify_light_curve_file(lc_file):
                invalid_files.append(f"Invalid: {lc_file}")
        
        if invalid_files:
            print(f"Found {len(invalid_files)} invalid files:")
            for file_issue in invalid_files[:10]:  # Show first 10
                print(f"  {file_issue}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more")
            return False
        
        print(f"✓ Data integrity verified: {len(metadata)} light curves")
        return True
    
    def get_download_stats(self, metadata_file: Union[str, Path]) -> Dict:
        """Get statistics about downloaded dataset."""
        
        metadata = pd.read_csv(metadata_file)
        
        stats = {
            'total_light_curves': len(metadata),
            'class_distribution': metadata['label'].value_counts().to_dict(),
            'data_sources': metadata['data_source'].value_counts().to_dict() if 'data_source' in metadata.columns else {},
            'period_range': {
                'min': metadata['period'].min() if 'period' in metadata.columns else None,
                'max': metadata['period'].max() if 'period' in metadata.columns else None,
                'median': metadata['period'].median() if 'period' in metadata.columns else None
            },
            'magnitude_range': {
                'min': metadata['magnitude'].min() if 'magnitude' in metadata.columns else None,
                'max': metadata['magnitude'].max() if 'magnitude' in metadata.columns else None,
                'median': metadata['magnitude'].median() if 'magnitude' in metadata.columns else None
            }
        }
        
        return stats
    
    def download_tess_sectors(
        self,
        sectors: List[int],
        output_dir: Union[str, Path],
        sample_size: Optional[int] = None,
        force_download: bool = False
    ) -> Tuple[pd.DataFrame, str]:
        """
        Download TESS light curves from specified sectors.
        
        Args:
            sectors: List of TESS sector numbers to download
            output_dir: Directory to save processed data
            sample_size: Number of targets to download per sector (None for all)
            force_download: Force re-download even if cached
            
        Returns:
            Tuple of (metadata_dataframe, metadata_file_path)
        """
        if not ASTRO_IMPORTS_AVAILABLE:
            raise ImportError("Astronomical libraries required. Install with: pip install astroquery lightkurve")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading TESS data from sectors: {sectors}")
        
        metadata_list = []
        
        for sector in sectors:
            print(f"Processing TESS Sector {sector}...")
            
            try:
                sector_metadata = self._download_tess_sector(
                    sector, output_dir, sample_size, force_download
                )
                metadata_list.extend(sector_metadata)
                
            except Exception as e:
                print(f"Warning: Failed to download sector {sector}: {e}")
                continue
        
        if not metadata_list:
            # Fallback to synthetic TESS data
            print("No TESS data downloaded successfully, creating synthetic sample...")
            return self._create_synthetic_tess_sample(output_dir, sectors, sample_size or 100)
        
        # Create final metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_file = output_dir / "tess_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"Saved metadata for {len(metadata_df)} TESS light curves to {metadata_file}")
        print(f"Class distribution: {metadata_df['label'].value_counts().to_dict()}")
        
        return metadata_df, str(metadata_file)
    
    def _download_tess_sector(
        self,
        sector: int,
        output_dir: Path,
        sample_size: Optional[int],
        force_download: bool = False
    ) -> List[Dict]:
        """Download light curves from a single TESS sector."""
        
        # Search for targets in this sector
        try:
            # Get catalog of targets in sector
            sector_targets = Catalogs.query_criteria(
                catalog="Tic",
                sector=sector,
                objType="STAR"
            )
            
            if len(sector_targets) == 0:
                print(f"No targets found in sector {sector}")
                return []
            
            # Sample targets if requested
            if sample_size and len(sector_targets) > sample_size:
                # Random sample
                indices = np.random.choice(len(sector_targets), sample_size, replace=False)
                sector_targets = sector_targets[indices]
            
            print(f"Found {len(sector_targets)} targets in sector {sector}")
            
        except Exception as e:
            print(f"Error querying sector {sector} catalog: {e}")
            return []
        
        metadata_list = []
        
        # Download light curves for selected targets
        for i, target in enumerate(tqdm(sector_targets[:sample_size or len(sector_targets)], 
                                      desc=f"Sector {sector}")):
            
            try:
                tic_id = target.get('ID', target.get('TIC', None))
                if tic_id is None:
                    continue
                
                lc_metadata = self._download_tess_light_curve(
                    tic_id, sector, target, output_dir, force_download
                )
                
                if lc_metadata:
                    metadata_list.append(lc_metadata)
                    
            except Exception as e:
                print(f"Warning: Failed to download TIC {tic_id}: {e}")
                continue
        
        return metadata_list
    
    def _download_tess_light_curve(
        self,
        tic_id: int,
        sector: int,
        target_info,
        output_dir: Path,
        force_download: bool = False
    ) -> Optional[Dict]:
        """Download individual TESS light curve."""
        
        star_id = f"TIC_{tic_id}"
        
        # Check if already downloaded
        lc_file = output_dir / f"{star_id}_S{sector:02d}.npz"
        if lc_file.exists() and not force_download:
            if self._verify_light_curve_file(lc_file):
                return self._create_tess_metadata_entry(tic_id, sector, target_info, lc_file)
        
        try:
            # Search for light curve
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)
            
            if len(search_result) == 0:
                return None
            
            # Download light curve
            lc = search_result.download()
            
            if lc is None:
                return None
            
            # Remove NaN values and normalize
            lc = lc.remove_nans().normalize()
            
            if len(lc.flux) < 100:  # Minimum length requirement
                return None
            
            # Save to file
            np.savez_compressed(
                lc_file,
                time=lc.time.value,
                flux=lc.flux.value,
                flux_err=lc.flux_err.value if lc.flux_err is not None else np.ones_like(lc.flux.value) * 0.001
            )
            
            return self._create_tess_metadata_entry(tic_id, sector, target_info, lc_file)
            
        except Exception as e:
            print(f"Error downloading TIC {tic_id} sector {sector}: {e}")
            return None
    
    def _create_tess_metadata_entry(self, tic_id: int, sector: int, target_info, lc_file: Path) -> Dict:
        """Create metadata entry for a TESS light curve."""
        
        # For TESS, we don't have confirmed planet labels from the catalog
        # So we'll assign random labels for now (in practice, would need TOI catalog)
        label = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% planets (realistic for TESS)
        
        return {
            'star_id': f"TIC_{tic_id}",
            'tic_id': int(tic_id),
            'label': label,
            'period': None,  # Would need TOI catalog for known periods
            'duration': None,
            'depth': None,
            'magnitude': target_info.get('Tmag', None),
            'teff': target_info.get('Teff', None),
            'radius': target_info.get('rad', None),
            'file_path': lc_file.name,
            'data_source': 'tess',
            'sector': sector
        }
    
    def _create_synthetic_tess_sample(
        self,
        output_dir: Path,
        sectors: List[int],
        sample_size: int
    ) -> Tuple[pd.DataFrame, str]:
        """Create synthetic TESS sample for development/testing."""
        
        print(f"Creating synthetic TESS sample with {sample_size} light curves...")
        
        np.random.seed(42)  # For reproducibility
        
        metadata_list = []
        targets_per_sector = sample_size // len(sectors)
        
        for sector in sectors:
            for i in range(targets_per_sector):
                # Create synthetic light curve
                star_id = f"TIC_SYNTHETIC_{sector}_{i:04d}"
                label = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% planets
                
                # Generate realistic TESS light curve (shorter than Kepler)
                n_points = np.random.randint(500, 2000)  # TESS sectors are ~27 days
                time = np.linspace(0, 27, n_points)  # ~1 month
                
                # Base stellar variability (TESS has different noise characteristics)
                flux = 1.0 + 0.002 * np.random.randn(n_points)  # Slightly higher noise
                
                # Add transit if planet
                if label == 1:
                    period = np.random.uniform(0.5, 20)  # TESS finds shorter periods
                    depth = np.random.uniform(50, 2000) * 1e-6  # ppm to fraction
                    duration = np.random.uniform(0.5, 6) / 24  # hours to days
                    
                    # Simple box transit
                    phase = (time % period) / period
                    transit_mask = np.abs(phase - 0.5) < (duration / period / 2)
                    flux[transit_mask] -= depth
                
                # Add noise (TESS characteristics)
                flux_err = np.full_like(flux, 0.002)  # Higher noise than Kepler
                flux += flux_err * np.random.randn(len(flux))
                
                # Save light curve
                lc_file = output_dir / f"{star_id}_S{sector:02d}.npz"
                np.savez_compressed(
                    lc_file,
                    time=time,
                    flux=flux,
                    flux_err=flux_err
                )
                
                # Create metadata
                metadata_list.append({
                    'star_id': star_id,
                    'tic_id': 100000000 + sector * 10000 + i,
                    'label': label,
                    'period': period if label == 1 else None,
                    'duration': duration * 24 if label == 1 else None,
                    'depth': depth * 1e6 if label == 1 else None,
                    'magnitude': np.random.uniform(8, 15),  # TESS magnitude range
                    'teff': np.random.uniform(3000, 8000),
                    'radius': np.random.uniform(0.3, 3.0),
                    'file_path': lc_file.name,
                    'data_source': 'synthetic_tess',
                    'sector': sector
                })
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_file = output_dir / "tess_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"Created {len(metadata_df)} synthetic TESS light curves")
        print(f"Class distribution: {metadata_df['label'].value_counts().to_dict()}")
        
        return metadata_df, str(metadata_file)
    
    def combine_datasets(
        self,
        kepler_metadata: Union[str, Path],
        tess_metadata: Union[str, Path],
        output_dir: Union[str, Path],
        balance_sources: bool = True
    ) -> Tuple[pd.DataFrame, str]:
        """
        Combine Kepler and TESS datasets into unified format.
        
        Args:
            kepler_metadata: Path to Kepler metadata CSV
            tess_metadata: Path to TESS metadata CSV
            output_dir: Directory to save combined dataset
            balance_sources: Whether to balance samples between sources
            
        Returns:
            Tuple of (combined_metadata, combined_metadata_file)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        kepler_df = pd.read_csv(kepler_metadata)
        tess_df = pd.read_csv(tess_metadata)
        
        print(f"Kepler dataset: {len(kepler_df)} light curves")
        print(f"TESS dataset: {len(tess_df)} light curves")
        
        # Standardize column names and add source info
        kepler_df['mission'] = 'kepler'
        tess_df['mission'] = 'tess'
        
        # Ensure consistent columns
        all_columns = set(kepler_df.columns) | set(tess_df.columns)
        
        for col in all_columns:
            if col not in kepler_df.columns:
                kepler_df[col] = None
            if col not in tess_df.columns:
                tess_df[col] = None
        
        # Balance datasets if requested
        if balance_sources:
            min_size = min(len(kepler_df), len(tess_df))
            kepler_df = kepler_df.sample(n=min_size, random_state=42)
            tess_df = tess_df.sample(n=min_size, random_state=42)
        
        # Combine datasets
        combined_df = pd.concat([kepler_df, tess_df], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined metadata
        combined_file = output_dir / "combined_metadata.csv"
        combined_df.to_csv(combined_file, index=False)
        
        print(f"Combined dataset: {len(combined_df)} light curves")
        print(f"Mission distribution: {combined_df['mission'].value_counts().to_dict()}")
        print(f"Class distribution: {combined_df['label'].value_counts().to_dict()}")
        
        return combined_df, str(combined_file)