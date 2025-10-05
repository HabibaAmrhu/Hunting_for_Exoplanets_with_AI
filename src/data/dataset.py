"""
PyTorch Dataset classes for exoplanet detection pipeline.
"""

import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .types import LightCurve, ProcessedLightCurve, PreprocessingConfig
# from .augmentation import AugmentationPipeline, create_standard_augmentation_pipeline


class LightCurveDataset(Dataset):
    """PyTorch dataset for light curve data with lazy loading and caching."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        preload: bool = False,
        max_cache_size: int = 1000
    ):
        """
        Initialize the light curve dataset.
        
        Args:
            data_path: Path to the data directory or file
            transform: Optional transform to apply to data
            cache_dir: Directory for caching processed data
            preload: Whether to preload all data into memory
            max_cache_size: Maximum number of items to keep in memory cache
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload = preload
        self.max_cache_size = max_cache_size
        
        # Initialize caches
        self._memory_cache: Dict[int, ProcessedLightCurve] = {}
        self._cache_access_order: List[int] = []
        
        # Load metadata and initialize dataset
        self._load_metadata()
        
        if self.preload:
            self._preload_data()
    
    def _load_metadata(self):
        """Load dataset metadata and file paths."""
        if self.data_path.is_file():
            # Single file dataset (e.g., .npz or .pkl)
            self._load_single_file_metadata()
        else:
            # Directory with multiple files
            self._load_directory_metadata()
    
    def _load_single_file_metadata(self):
        """Load metadata from a single file."""
        if self.data_path.suffix == '.npz':
            data = np.load(self.data_path, allow_pickle=True)
            self.star_ids = data['star_ids']
            self.labels = data['labels']
            self.metadata = data.get('metadata', [{}] * len(self.star_ids))
            self.file_paths = [self.data_path] * len(self.star_ids)
            self.indices = list(range(len(self.star_ids)))
        elif self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            self.star_ids = data['star_ids']
            self.labels = data['labels']
            self.metadata = data.get('metadata', [{}] * len(self.star_ids))
            self.file_paths = [self.data_path] * len(self.star_ids)
            self.indices = list(range(len(self.star_ids)))
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _load_directory_metadata(self):
        """Load metadata from a directory of files."""
        # Look for metadata file
        metadata_file = self.data_path / 'metadata.csv'
        if metadata_file.exists():
            df = pd.read_csv(metadata_file)
            self.star_ids = df['star_id'].values
            self.labels = df['label'].values
            self.metadata = df.to_dict('records')
            
            # Find corresponding data files
            self.file_paths = []
            self.indices = []
            
            for i, star_id in enumerate(self.star_ids):
                data_file = self.data_path / f"{star_id}.npz"
                if data_file.exists():
                    self.file_paths.append(data_file)
                    self.indices.append(i)
        else:
            # Scan directory for .npz files
            data_files = list(self.data_path.glob("*.npz"))
            self.star_ids = [f.stem for f in data_files]
            self.labels = [0] * len(data_files)  # Default to no planet
            self.metadata = [{}] * len(data_files)
            self.file_paths = data_files
            self.indices = list(range(len(data_files)))
    
    def _preload_data(self):
        """Preload all data into memory."""
        print(f"Preloading {len(self)} samples...")
        for i in range(len(self)):
            self._load_item(i)
        print("Preloading complete!")
    
    def _load_item(self, idx: int) -> ProcessedLightCurve:
        """Load a single item from disk."""
        # Check memory cache first
        if idx in self._memory_cache:
            self._update_cache_access(idx)
            return self._memory_cache[idx]
        
        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"item_{idx}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        item = pickle.load(f)
                    self._add_to_memory_cache(idx, item)
                    return item
                except Exception:
                    pass  # Fall back to loading from source
        
        # Load from source
        file_path = self.file_paths[idx]
        data_idx = self.indices[idx]
        
        if file_path.suffix == '.npz':
            data = np.load(file_path, allow_pickle=True)
            if len(data['light_curves']) > data_idx:
                light_curve_data = data['light_curves'][data_idx]
            else:
                light_curve_data = data['light_curves'][0]  # Fallback
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Create ProcessedLightCurve object
        item = ProcessedLightCurve(
            star_id=self.star_ids[idx],
            flux=light_curve_data,
            time=np.arange(len(light_curve_data[0])),  # Default time array
            label=self.labels[idx],
            metadata=self.metadata[idx]
        )
        
        # Cache the item
        if self.cache_dir:
            self._save_to_disk_cache(idx, item)
        
        self._add_to_memory_cache(idx, item)
        
        return item
    
    def _add_to_memory_cache(self, idx: int, item: ProcessedLightCurve):
        """Add item to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self.max_cache_size:
            # Remove least recently used item
            lru_idx = self._cache_access_order.pop(0)
            del self._memory_cache[lru_idx]
        
        self._memory_cache[idx] = item
        self._cache_access_order.append(idx)
    
    def _update_cache_access(self, idx: int):
        """Update cache access order for LRU."""
        if idx in self._cache_access_order:
            self._cache_access_order.remove(idx)
        self._cache_access_order.append(idx)
    
    def _save_to_disk_cache(self, idx: int, item: ProcessedLightCurve):
        """Save item to disk cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"item_{idx}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(item, f)
        except Exception as e:
            print(f"Warning: Could not save to disk cache: {e}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.star_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, label, metadata)
        """
        item = self._load_item(idx)
        
        # Convert to tensor
        data = torch.tensor(item.flux, dtype=torch.float32)
        label = torch.tensor(item.label, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)
        
        return data, label, item.metadata
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced training."""
        class_counts = self.get_class_distribution()
        total_samples = len(self)
        
        # Calculate weights inversely proportional to class frequency
        weights = []
        for label in self.labels:
            weight = total_samples / (len(class_counts) * class_counts[label])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class AugmentedLightCurveDataset(LightCurveDataset):
    """Dataset with integrated augmentation pipeline."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        augmentation_pipeline: Optional[object] = None,
        augmentation_probability: float = 0.5,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        preload: bool = False,
        max_cache_size: int = 1000
    ):
        """
        Initialize augmented dataset.
        
        Args:
            data_path: Path to the data directory or file
            augmentation_pipeline: Augmentation pipeline to apply
            augmentation_probability: Probability of applying augmentation
            transform: Optional transform to apply to data
            cache_dir: Directory for caching processed data
            preload: Whether to preload all data into memory
            max_cache_size: Maximum number of items to keep in memory cache
        """
        super().__init__(data_path, transform, cache_dir, preload, max_cache_size)
        
        self.augmentation_pipeline = augmentation_pipeline or create_standard_augmentation_pipeline()
        self.augmentation_probability = augmentation_probability
        self.training_mode = True  # Enable/disable augmentation
    
    def set_training_mode(self, training: bool):
        """Set training mode to enable/disable augmentation."""
        self.training_mode = training
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample with optional augmentation.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, label, metadata)
        """
        item = self._load_item(idx)
        
        # Get data as numpy array
        data = item.flux.copy()
        metadata = item.metadata.copy()
        
        # Apply augmentation if in training mode
        if self.training_mode and np.random.random() < self.augmentation_probability:
            data, metadata = self.augmentation_pipeline(data, metadata)
        
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(item.label, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)
        
        return data, label, metadata


class SyntheticAugmentedDataset(Dataset):
    """Dataset that combines real data with synthetic transit injection."""
    
    def __init__(
        self,
        real_dataset: LightCurveDataset,
        synthetic_injector: Optional[Callable] = None,
        synthetic_ratio: float = 0.3,
        augmentation_pipeline: Optional[object] = None
    ):
        """
        Initialize synthetic augmented dataset.
        
        Args:
            real_dataset: Base dataset with real light curves
            synthetic_injector: Function to inject synthetic transits
            synthetic_ratio: Ratio of synthetic to real samples
            augmentation_pipeline: Additional augmentation pipeline
        """
        self.real_dataset = real_dataset
        self.synthetic_injector = synthetic_injector
        self.synthetic_ratio = synthetic_ratio
        self.augmentation_pipeline = augmentation_pipeline
        
        # Calculate dataset sizes
        self.real_size = len(real_dataset)
        self.synthetic_size = int(self.real_size * synthetic_ratio)
        self.total_size = self.real_size + self.synthetic_size
        
        # Pre-generate synthetic samples for consistency
        self._generate_synthetic_samples()
    
    def _generate_synthetic_samples(self):
        """Pre-generate synthetic samples."""
        if not self.synthetic_injector:
            self.synthetic_samples = []
            return
        
        self.synthetic_samples = []
        
        # Generate synthetic samples by injecting transits into real light curves
        for i in range(self.synthetic_size):
            # Select random real sample as base
            base_idx = np.random.randint(0, self.real_size)
            base_data, _, base_metadata = self.real_dataset[base_idx]
            
            # Inject synthetic transit
            synthetic_data, synthetic_metadata = self.synthetic_injector(
                base_data.numpy(), base_metadata
            )
            
            # Store synthetic sample
            self.synthetic_samples.append({
                'data': synthetic_data,
                'label': 1,  # Synthetic samples have planets
                'metadata': synthetic_metadata
            })
    
    def __len__(self) -> int:
        """Return total dataset size."""
        return self.total_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get sample from combined real and synthetic dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, label, metadata)
        """
        if idx < self.real_size:
            # Return real sample
            data, label, metadata = self.real_dataset[idx]
        else:
            # Return synthetic sample
            synthetic_idx = idx - self.real_size
            sample = self.synthetic_samples[synthetic_idx]
            
            data = torch.tensor(sample['data'], dtype=torch.float32)
            label = torch.tensor(sample['label'], dtype=torch.float32)
            metadata = sample['metadata']
        
        # Apply additional augmentation if provided
        if self.augmentation_pipeline:
            data_np, metadata = self.augmentation_pipeline(data.numpy(), metadata)
            data = torch.tensor(data_np, dtype=torch.float32)
        
        return data, label, metadata
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution including synthetic samples."""
        real_dist = self.real_dataset.get_class_distribution()
        
        # Add synthetic samples (all positive)
        total_dist = real_dist.copy()
        total_dist[1] = total_dist.get(1, 0) + self.synthetic_size
        
        return total_dist


def create_train_val_datasets(
    data_path: Union[str, Path],
    val_split: float = 0.2,
    use_augmentation: bool = True,
    synthetic_injector: Optional[Callable] = None,
    synthetic_ratio: float = 0.3,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Create training and validation datasets with optional augmentation.
    
    Args:
        data_path: Path to the data
        val_split: Fraction of data to use for validation
        use_augmentation: Whether to use augmentation for training
        synthetic_injector: Optional synthetic transit injector
        synthetic_ratio: Ratio of synthetic to real samples
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Load base dataset
    full_dataset = LightCurveDataset(data_path)
    
    # Create train/val split
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Random split
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Add augmentation to training dataset
    if use_augmentation:
        if synthetic_injector:
            # Use synthetic augmentation
            train_dataset = SyntheticAugmentedDataset(
                train_dataset,
                synthetic_injector=synthetic_injector,
                synthetic_ratio=synthetic_ratio,
                augmentation_pipeline=create_standard_augmentation_pipeline()
            )
        else:
            # Use traditional augmentation
            train_dataset = AugmentedLightCurveDataset(
                data_path,
                augmentation_pipeline=create_standard_augmentation_pipeline()
            )
            # Apply subset indices
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    return train_dataset, val_dataset


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function for light curve data.
    
    Args:
        batch: List of (data, label, metadata) tuples
        
    Returns:
        Batched tensors and metadata list
    """
    data_list, label_list, metadata_list = zip(*batch)
    
    # Stack tensors
    data_batch = torch.stack(data_list)
    label_batch = torch.stack(label_list)
    
    return data_batch, label_batch, list(metadata_list)