"""
Pytest configuration and shared fixtures for the exoplanet detection pipeline tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def device():
    """Provide device for testing (CPU or CUDA if available)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture(scope="session")
def temp_dir():
    """Provide temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_light_curve_data():
    """Provide sample light curve data for testing."""
    np.random.seed(42)
    
    n_samples = 20
    n_channels = 2
    sequence_length = 512
    
    data = np.random.randn(n_samples, n_channels, sequence_length)
    labels = np.random.randint(0, 2, n_samples)
    metadata = [{'star_id': f'test_{i}'} for i in range(n_samples)]
    
    return data, labels, metadata


@pytest.fixture
def sample_transit_data():
    """Provide sample data with synthetic transits."""
    np.random.seed(42)
    
    sequence_length = 512
    n_channels = 2
    
    # Create base stellar signal
    time = np.arange(sequence_length)
    flux = np.ones(sequence_length) + 0.01 * np.random.randn(sequence_length)
    
    # Add transit signal
    transit_center = sequence_length // 2
    transit_width = 30
    transit_depth = 0.02
    
    start = max(0, transit_center - transit_width // 2)
    end = min(sequence_length, transit_center + transit_width // 2)
    flux[start:end] -= transit_depth
    
    # Create dual-channel data
    raw_channel = flux
    phase_folded_channel = flux + 0.005 * np.random.randn(sequence_length)
    
    data = np.stack([raw_channel, phase_folded_channel])
    
    return data, {'star_id': 'transit_test', 'has_transit': True}


@pytest.fixture
def mock_stellar_parameters():
    """Provide mock stellar parameters for testing."""
    return {
        'temperature': 5778,
        'radius': 1.0,
        'mass': 1.0,
        'magnitude': 12.0,
        'metallicity': 0.0
    }


@pytest.fixture
def mock_transit_parameters():
    """Provide mock transit parameters for testing."""
    return {
        'period': 10.0,
        'radius_ratio': 0.1,
        'impact_parameter': 0.3,
        'limb_darkening': [0.3, 0.2],
        'eccentricity': 0.0,
        'omega': 0.0
    }


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration parameters."""
    return {
        'batch_size': 4,
        'sequence_length': 512,
        'n_channels': 2,
        'learning_rate': 0.001,
        'epochs': 2,
        'patience': 5
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "integration" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during testing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield