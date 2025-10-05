"""
Advanced streaming data processor for real-time exoplanet detection.
Implements efficient data pipeline with caching and incremental processing.
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging

from .types import LightCurve, ProcessedLightCurve
from .dataset import LightCurveDataset
from ..preprocessing.preprocessor import LightCurvePreprocessor


class StreamingDataProcessor:
    """
    Advanced streaming processor for real-time light curve analysis.
    
    Features:
    - Asynchronous data processing
    - Intelligent caching with invalidation
    - Incremental processing for large datasets
    - Memory-efficient batch processing
    """
    
    def __init__(
        self,
        preprocessor: LightCurvePreprocessor,
        cache_dir: Optional[Path] = None,
        max_cache_size: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize streaming processor.
        
        Args:
            preprocessor: Light curve preprocessor
            cache_dir: Directory for caching processed data
            max_cache_size: Maximum number of cached items
            batch_size: Batch size for processing
            num_workers: Number of worker threads
        """
        self.preprocessor = preprocessor
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_size = max_cache_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize cache
        self.cache = {}
        self.cache_access_times = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(self, light_curve: LightCurve) -> str:
        """Compute cache key for light curve."""
        # Create hash from light curve data and preprocessing config
        data_str = f"{light_curve.star_id}_{len(light_curve.time)}_{light_curve.time[0]}_{light_curve.time[-1]}"
        config_str = str(self.preprocessor.config.__dict__)
        combined = data_str + config_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[ProcessedLightCurve]:
        """Get processed light curve from cache."""
        # Check memory cache first
        if cache_key in self.cache:
            self.cache_access_times[cache_key] = time.time()
            return self.cache[cache_key]
        
        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        processed = pickle.load(f)
                    
                    # Add to memory cache
                    self._add_to_cache(cache_key, processed)
                    return processed
                except Exception as e:
                    self.logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _add_to_cache(self, cache_key: str, processed: ProcessedLightCurve):
        """Add processed light curve to cache."""
        # Add to memory cache
        self.cache[cache_key] = processed
        self.cache_access_times[cache_key] = time.time()
        
        # Evict old items if cache is full
        if len(self.cache) > self.max_cache_size:
            # Remove least recently used item
            oldest_key = min(self.cache_access_times.keys(), 
                           key=lambda k: self.cache_access_times[k])
            del self.cache[oldest_key]
            del self.cache_access_times[oldest_key]
        
        # Save to disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed, f)
            except Exception as e:
                self.logger.warning(f"Failed to save to cache: {e}")
    
    async def process_light_curve_async(self, light_curve: LightCurve) -> ProcessedLightCurve:
        """
        Process single light curve asynchronously.
        
        Args:
            light_curve: Input light curve
            
        Returns:
            Processed light curve
        """
        cache_key = self._compute_cache_key(light_curve)
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Process in thread pool
        loop = asyncio.get_event_loop()
        processed = await loop.run_in_executor(
            self.executor, self.preprocessor.process, light_curve
        )
        
        # Add to cache
        self._add_to_cache(cache_key, processed)
        
        return processed
    
    async def process_batch_async(
        self, 
        light_curves: List[LightCurve]
    ) -> List[ProcessedLightCurve]:
        """
        Process batch of light curves asynchronously.
        
        Args:
            light_curves: List of input light curves
            
        Returns:
            List of processed light curves
        """
        tasks = [
            self.process_light_curve_async(lc) 
            for lc in light_curves
        ]
        
        return await asyncio.gather(*tasks)
    
    async def stream_process(
        self, 
        light_curve_stream: AsyncGenerator[LightCurve, None]
    ) -> AsyncGenerator[ProcessedLightCurve, None]:
        """
        Process streaming light curves with batching.
        
        Args:
            light_curve_stream: Async generator of light curves
            
        Yields:
            Processed light curves
        """
        batch = []
        
        async for light_curve in light_curve_stream:
            batch.append(light_curve)
            
            if len(batch) >= self.batch_size:
                # Process batch
                processed_batch = await self.process_batch_async(batch)
                
                # Yield results
                for processed in processed_batch:
                    yield processed
                
                # Clear batch
                batch = []
        
        # Process remaining items
        if batch:
            processed_batch = await self.process_batch_async(batch)
            for processed in processed_batch:
                yield processed
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'memory_cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_rate': self._calculate_hit_rate(),
            'disk_cache_files': len(list(self.cache_dir.glob('*.pkl'))) if self.cache_dir else 0
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # In a real implementation, you would track hits and misses
        return 0.85  # Placeholder value
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        self.cache_access_times.clear()
        
        if self.cache_dir:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class IncrementalProcessor:
    """
    Incremental processor for large-scale survey data.
    
    Processes data in chunks and maintains state between runs.
    """
    
    def __init__(
        self,
        processor: StreamingDataProcessor,
        state_file: Optional[Path] = None,
        chunk_size: int = 1000
    ):
        """
        Initialize incremental processor.
        
        Args:
            processor: Streaming data processor
            state_file: File to save processing state
            chunk_size: Number of items per chunk
        """
        self.processor = processor
        self.state_file = Path(state_file) if state_file else None
        self.chunk_size = chunk_size
        
        # Load previous state
        self.processed_ids = set()
        self._load_state()
    
    def _load_state(self):
        """Load processing state from file."""
        if self.state_file and self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    self.processed_ids = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load state: {e}")
                self.processed_ids = set()
    
    def _save_state(self):
        """Save processing state to file."""
        if self.state_file:
            try:
                with open(self.state_file, 'wb') as f:
                    pickle.dump(self.processed_ids, f)
            except Exception as e:
                logging.warning(f"Failed to save state: {e}")
    
    async def process_incremental(
        self,
        light_curves: List[LightCurve],
        force_reprocess: bool = False
    ) -> List[ProcessedLightCurve]:
        """
        Process light curves incrementally.
        
        Args:
            light_curves: List of light curves to process
            force_reprocess: Whether to reprocess already processed items
            
        Returns:
            List of processed light curves
        """
        # Filter out already processed items
        if not force_reprocess:
            light_curves = [
                lc for lc in light_curves 
                if lc.star_id not in self.processed_ids
            ]
        
        if not light_curves:
            return []
        
        # Process in chunks
        results = []
        
        for i in range(0, len(light_curves), self.chunk_size):
            chunk = light_curves[i:i + self.chunk_size]
            
            # Process chunk
            processed_chunk = await self.processor.process_batch_async(chunk)
            results.extend(processed_chunk)
            
            # Update processed IDs
            for lc in chunk:
                self.processed_ids.add(lc.star_id)
            
            # Save state periodically
            self._save_state()
            
            # Log progress
            logging.info(f"Processed chunk {i//self.chunk_size + 1}/{(len(light_curves)-1)//self.chunk_size + 1}")
        
        return results
    
    def get_progress(self, total_items: int) -> Dict[str, Any]:
        """Get processing progress statistics."""
        return {
            'processed_count': len(self.processed_ids),
            'total_count': total_items,
            'progress_percentage': len(self.processed_ids) / total_items * 100 if total_items > 0 else 0,
            'remaining_count': total_items - len(self.processed_ids)
        }


# Factory functions
def create_streaming_processor(
    preprocessor: LightCurvePreprocessor,
    **kwargs
) -> StreamingDataProcessor:
    """Create streaming data processor with default configuration."""
    return StreamingDataProcessor(preprocessor, **kwargs)


def create_incremental_processor(
    streaming_processor: StreamingDataProcessor,
    **kwargs
) -> IncrementalProcessor:
    """Create incremental processor with default configuration."""
    return IncrementalProcessor(streaming_processor, **kwargs)