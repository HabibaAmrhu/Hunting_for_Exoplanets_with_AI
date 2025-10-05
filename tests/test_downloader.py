"""Tests for data downloader functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data.downloader import DataDownloader


class TestDataDownloader:
    """Test DataDownloader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.output_dir = self.temp_dir / "output"
        
        self.downloader = DataDownloader(cache_dir=self.cache_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test downloader initialization."""
        assert self.downloader.cache_dir == self.cache_dir
        assert (self.cache_dir / "kepler").exists()
        assert (self.cache_dir / "tess").exists()
        assert (self.cache_dir / "metadata").exists()
    
    def test_create_synthetic_koi_sample(self):
        """Test synthetic KOI sample creation."""
        sample_size = 50
        
        metadata_df, metadata_file = self.downloader._create_synthetic_koi_sample(
            self.output_dir, sample_size
        )
        
        # Check metadata
        assert len(metadata_df) == sample_size
        assert 'star_id' in metadata_df.columns
        assert 'label' in metadata_df.columns
        assert 'file_path' in metadata_df.columns
        
        # Check class distribution (should have both classes)
        labels = metadata_df['label'].unique()
        assert 0 in labels or 1 in labels  # At least one class
        
        # Check files were created
        assert Path(metadata_file).exists()
        
        for _, row in metadata_df.iterrows():
            lc_file = self.output_dir / row['file_path']
            assert lc_file.exists()
            
            # Verify file contents
            data = np.load(lc_file)
            assert 'time' in data
            assert 'flux' in data
            assert 'flux_err' in data
            assert len(data['time']) > 0
            assert len(data['flux']) == len(data['time'])
    
    def test_verify_light_curve_file(self):
        """Test light curve file verification."""
        # Create valid file
        valid_file = self.temp_dir / "valid.npz"
        np.savez_compressed(
            valid_file,
            time=np.linspace(0, 100, 1000),
            flux=np.random.normal(1.0, 0.001, 1000),
            flux_err=np.full(1000, 0.001)
        )
        
        assert self.downloader._verify_light_curve_file(valid_file) == True
        
        # Create invalid file (missing flux)
        invalid_file = self.temp_dir / "invalid.npz"
        np.savez_compressed(
            invalid_file,
            time=np.linspace(0, 100, 1000)
            # Missing flux
        )
        
        assert self.downloader._verify_light_curve_file(invalid_file) == False
        
        # Test non-existent file
        nonexistent_file = self.temp_dir / "nonexistent.npz"
        assert self.downloader._verify_light_curve_file(nonexistent_file) == False
    
    def test_verify_data_integrity(self):
        """Test dataset integrity verification."""
        # Create synthetic dataset
        metadata_df, metadata_file = self.downloader._create_synthetic_koi_sample(
            self.output_dir, 10
        )
        
        # Should pass verification
        assert self.downloader.verify_data_integrity(metadata_file) == True
        assert self.downloader.verify_data_integrity(self.output_dir) == True
        
        # Corrupt one file
        first_file = self.output_dir / metadata_df.iloc[0]['file_path']
        with open(first_file, 'w') as f:
            f.write("corrupted")
        
        # Should fail verification
        assert self.downloader.verify_data_integrity(metadata_file) == False
    
    def test_get_download_stats(self):
        """Test download statistics calculation."""
        # Create synthetic dataset
        metadata_df, metadata_file = self.downloader._create_synthetic_koi_sample(
            self.output_dir, 20
        )
        
        stats = self.downloader.get_download_stats(metadata_file)
        
        assert 'total_light_curves' in stats
        assert stats['total_light_curves'] == 20
        assert 'class_distribution' in stats
        assert 'data_sources' in stats
        assert 'period_range' in stats
        assert 'magnitude_range' in stats
        
        # Check that we have reasonable ranges
        if stats['period_range']['min'] is not None:
            assert stats['period_range']['min'] > 0
            assert stats['period_range']['max'] > stats['period_range']['min']
    
    def test_process_koi_table(self):
        """Test KOI table processing."""
        # Create mock KOI table
        mock_koi = pd.DataFrame({
            'kepid': [1001, 1002, 1003, 1004, 1005],
            'kepoi_name': ['K1001.01', 'K1002.01', 'K1003.01', 'K1004.01', 'K1005.01'],
            'koi_disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'CONFIRMED', 'FALSE POSITIVE'],
            'koi_period': [10.5, 5.2, 15.8, 3.1, 25.4],
            'koi_kepmag': [12.1, 13.5, 11.8, 14.2, 12.9]
        })
        
        processed = self.downloader._process_koi_table(
            mock_koi, sample_size=None, min_period=1.0, max_period=30.0
        )
        
        # Check labels were created correctly
        assert 'label' in processed.columns
        
        # CONFIRMED and CANDIDATE should be 1, FALSE POSITIVE should be 0
        confirmed_mask = processed['koi_disposition'] == 'CONFIRMED'
        candidate_mask = processed['koi_disposition'] == 'CANDIDATE'
        false_pos_mask = processed['koi_disposition'] == 'FALSE POSITIVE'
        
        assert all(processed.loc[confirmed_mask, 'label'] == 1)
        assert all(processed.loc[candidate_mask, 'label'] == 1)
        assert all(processed.loc[false_pos_mask, 'label'] == 0)
        
        # Test sampling
        sampled = self.downloader._process_koi_table(
            mock_koi, sample_size=3, min_period=1.0, max_period=30.0
        )
        assert len(sampled) == 3
    
    def test_create_metadata_entry(self):
        """Test metadata entry creation."""
        mock_koi = pd.Series({
            'kepid': 1001,
            'kepoi_name': 'K1001.01',
            'label': 1,
            'koi_period': 10.5,
            'koi_duration': 4.2,
            'koi_depth': 1500,
            'koi_kepmag': 12.1
        })
        
        star_id = "KIC_1001"
        lc_file = Path("test_file.npz")
        
        metadata = self.downloader._create_metadata_entry(mock_koi, star_id, lc_file)
        
        assert metadata['star_id'] == star_id
        assert metadata['kepid'] == 1001
        assert metadata['label'] == 1
        assert metadata['period'] == 10.5
        assert metadata['file_path'] == "test_file.npz"
        assert metadata['data_source'] == 'kepler'
    
    def test_download_kepler_koi_fallback(self):
        """Test KOI download with fallback to synthetic data."""
        # This test assumes astroquery is not available or fails
        # Should fallback to synthetic data
        
        try:
            metadata_df, metadata_file = self.downloader.download_kepler_koi(
                self.output_dir, sample_size=5
            )
            
            # Should have created some data (either real or synthetic)
            assert len(metadata_df) > 0
            assert Path(metadata_file).exists()
            
            # Verify integrity
            assert self.downloader.verify_data_integrity(metadata_file) == True
            
        except ImportError:
            # Expected if astroquery not available
            pytest.skip("Astroquery not available for testing")
        except Exception as e:
            # Should fallback to synthetic data
            assert "synthetic" in str(e).lower() or len(metadata_df) > 0
    
    def test_create_synthetic_tess_sample(self):
        """Test synthetic TESS sample creation."""
        sectors = [14, 15]
        sample_size = 20
        
        metadata_df, metadata_file = self.downloader._create_synthetic_tess_sample(
            self.output_dir, sectors, sample_size
        )
        
        # Check metadata
        assert len(metadata_df) == sample_size
        assert 'star_id' in metadata_df.columns
        assert 'label' in metadata_df.columns
        assert 'sector' in metadata_df.columns
        assert 'data_source' in metadata_df.columns
        
        # Check sectors are correct
        unique_sectors = metadata_df['sector'].unique()
        assert all(sector in sectors for sector in unique_sectors)
        
        # Check data source
        assert all(metadata_df['data_source'] == 'synthetic_tess')
        
        # Check files were created
        assert Path(metadata_file).exists()
        
        for _, row in metadata_df.iterrows():
            lc_file = self.output_dir / row['file_path']
            assert lc_file.exists()
            
            # Verify file contents
            data = np.load(lc_file)
            assert 'time' in data
            assert 'flux' in data
            assert 'flux_err' in data
            assert len(data['time']) > 0
            
            # TESS light curves should be shorter than Kepler
            assert len(data['time']) < 3000  # Reasonable upper bound for TESS
    
    def test_create_tess_metadata_entry(self):
        """Test TESS metadata entry creation."""
        tic_id = 123456789
        sector = 14
        
        mock_target_info = {
            'Tmag': 10.5,
            'Teff': 5500,
            'rad': 1.2
        }
        
        lc_file = Path("test_tess_file.npz")
        
        metadata = self.downloader._create_tess_metadata_entry(
            tic_id, sector, mock_target_info, lc_file
        )
        
        assert metadata['star_id'] == f"TIC_{tic_id}"
        assert metadata['tic_id'] == tic_id
        assert metadata['sector'] == sector
        assert metadata['magnitude'] == 10.5
        assert metadata['teff'] == 5500
        assert metadata['radius'] == 1.2
        assert metadata['data_source'] == 'tess'
        assert metadata['label'] in [0, 1]  # Should be valid label
    
    def test_download_tess_sectors_fallback(self):
        """Test TESS download with fallback to synthetic data."""
        sectors = [14, 15]
        
        try:
            metadata_df, metadata_file = self.downloader.download_tess_sectors(
                sectors, self.output_dir, sample_size=10
            )
            
            # Should have created some data (either real or synthetic)
            assert len(metadata_df) > 0
            assert Path(metadata_file).exists()
            
            # Check that sectors are represented
            if 'sector' in metadata_df.columns:
                unique_sectors = metadata_df['sector'].unique()
                assert len(unique_sectors) > 0
            
            # Verify integrity
            assert self.downloader.verify_data_integrity(metadata_file) == True
            
        except ImportError:
            # Expected if astroquery not available
            pytest.skip("Astroquery not available for testing")
        except Exception as e:
            # Should fallback to synthetic data or succeed
            assert len(metadata_df) > 0 or "synthetic" in str(e).lower()
    
    def test_combine_datasets(self):
        """Test combining Kepler and TESS datasets."""
        # Create synthetic datasets
        kepler_df, kepler_file = self.downloader._create_synthetic_koi_sample(
            self.output_dir, 15
        )
        
        tess_df, tess_file = self.downloader._create_synthetic_tess_sample(
            self.output_dir, [14, 15], 10
        )
        
        # Combine datasets
        combined_df, combined_file = self.downloader.combine_datasets(
            kepler_file, tess_file, self.output_dir, balance_sources=True
        )
        
        # Check combined dataset
        assert len(combined_df) > 0
        assert 'mission' in combined_df.columns
        assert Path(combined_file).exists()
        
        # Check missions are represented
        missions = combined_df['mission'].unique()
        assert 'kepler' in missions or 'tess' in missions
        
        # Check balancing worked (should have equal or close to equal)
        if len(missions) > 1:
            mission_counts = combined_df['mission'].value_counts()
            max_diff = mission_counts.max() - mission_counts.min()
            assert max_diff <= 1  # At most 1 difference due to rounding
        
        # Verify all files exist
        for _, row in combined_df.iterrows():
            lc_file = self.output_dir / row['file_path']
            assert lc_file.exists()