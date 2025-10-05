"""
Load testing suite for the exoplanet detection pipeline.
Tests system behavior under various load conditions.
"""

import pytest
import time
import threading
import concurrent.futures
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import statistics
import json
from pathlib import Path
import psutil
import gc

from src.models import ExoplanetCNN
from src.inference import ModelInference
from src.api.server import app
from src.monitoring import ModelMonitor


class LoadTestResults:
    """Container for load test results."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.errors: List[str] = []
        self.throughput: float = 0.0
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
    
    def add_response(self, response_time: float, success: bool, error: str = None):
        """Add a response result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.response_times:
            return {}
        
        return {
            'total_requests': len(self.response_times),
            'success_rate': self.success_count / len(self.response_times),
            'error_rate': self.error_count / len(self.response_times),
            'avg_response_time': statistics.mean(self.response_times),
            'median_response_time': statistics.median(self.response_times),
            'p95_response_time': np.percentile(self.response_times, 95),
            'p99_response_time': np.percentile(self.response_times, 99),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'throughput': self.throughput,
            'avg_memory_usage': statistics.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage': max(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_usage': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu_usage': max(self.cpu_usage) if self.cpu_usage else 0
        }


class ModelLoadTester:
    """Load tester for model inference."""
    
    def __init__(self, model_path: str = None):
        """Initialize load tester."""
        if model_path:
            self.model = ModelInference(model_path)
        else:
            # Create a simple model for testing
            self.model = ExoplanetCNN(input_length=1000)
            self.model.eval()
    
    def single_prediction(self, data: torch.Tensor) -> Tuple[float, bool, str]:
        """Make a single prediction and measure time."""
        start_time = time.time()
        try:
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(data)
            else:
                with torch.no_grad():
                    prediction = self.model(data)
            
            end_time = time.time()
            return end_time - start_time, True, None
        except Exception as e:
            end_time = time.time()
            return end_time - start_time, False, str(e)
    
    def concurrent_load_test(
        self,
        num_requests: int,
        num_threads: int,
        data_shape: Tuple[int, ...] = (1, 1, 1000)
    ) -> LoadTestResults:
        """Run concurrent load test."""
        results = LoadTestResults()
        
        def worker():
            """Worker function for concurrent requests."""
            data = torch.randn(*data_shape)
            response_time, success, error = self.single_prediction(data)
            results.add_response(response_time, success, error)
        
        # Monitor system resources
        def monitor_resources():
            """Monitor system resources during test."""
            process = psutil.Process()
            while not stop_monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    results.memory_usage.append(memory_mb)
                    results.cpu_usage.append(cpu_percent)
                    time.sleep(0.1)
                except:
                    pass
        
        # Start resource monitoring
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run load test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_requests)]
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        # Calculate throughput
        total_time = end_time - start_time
        results.throughput = num_requests / total_time
        
        return results
    
    def sustained_load_test(
        self,
        duration_seconds: int,
        requests_per_second: int,
        data_shape: Tuple[int, ...] = (1, 1, 1000)
    ) -> LoadTestResults:
        """Run sustained load test for specified duration."""
        results = LoadTestResults()
        
        def worker():
            """Worker function for sustained load."""
            data = torch.randn(*data_shape)
            response_time, success, error = self.single_prediction(data)
            results.add_response(response_time, success, error)
        
        # Calculate request interval
        request_interval = 1.0 / requests_per_second
        
        # Run sustained load
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_second * 2) as executor:
            while time.time() < end_time:
                executor.submit(worker)
                time.sleep(request_interval)
        
        # Calculate throughput
        total_time = time.time() - start_time
        results.throughput = len(results.response_times) / total_time
        
        return results


class APILoadTester:
    """Load tester for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API load tester."""
        self.base_url = base_url
    
    def single_request(self, endpoint: str, data: Dict[str, Any]) -> Tuple[float, bool, str]:
        """Make a single API request and measure time."""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                return end_time - start_time, True, None
            else:
                return end_time - start_time, False, f"HTTP {response.status_code}"
        
        except Exception as e:
            end_time = time.time()
            return end_time - start_time, False, str(e)
    
    def concurrent_api_test(
        self,
        endpoint: str,
        num_requests: int,
        num_threads: int,
        payload_generator: callable
    ) -> LoadTestResults:
        """Run concurrent API load test."""
        results = LoadTestResults()
        
        def worker():
            """Worker function for API requests."""
            payload = payload_generator()
            response_time, success, error = self.single_request(endpoint, payload)
            results.add_response(response_time, success, error)
        
        # Run concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_requests)]
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        
        # Calculate throughput
        total_time = end_time - start_time
        results.throughput = num_requests / total_time
        
        return results


class TestModelLoadPerformance:
    """Test model performance under load."""
    
    def test_single_thread_performance(self):
        """Test single-threaded model performance."""
        tester = ModelLoadTester()
        
        # Test with varying batch sizes
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            results = tester.concurrent_load_test(
                num_requests=100,
                num_threads=1,
                data_shape=(batch_size, 1, 1000)
            )
            
            stats = results.calculate_statistics()
            
            # Assertions for performance
            assert stats['success_rate'] >= 0.95, f"Success rate too low for batch size {batch_size}: {stats['success_rate']}"
            assert stats['avg_response_time'] < 5.0, f"Average response time too high for batch size {batch_size}: {stats['avg_response_time']}"
            
            print(f"Batch size {batch_size}: {stats['avg_response_time']:.3f}s avg, {stats['throughput']:.1f} req/s")
    
    def test_concurrent_performance(self):
        """Test concurrent model performance."""
        tester = ModelLoadTester()
        
        # Test with varying thread counts
        thread_counts = [2, 4, 8]
        
        for num_threads in thread_counts:
            results = tester.concurrent_load_test(
                num_requests=200,
                num_threads=num_threads,
                data_shape=(1, 1, 1000)
            )
            
            stats = results.calculate_statistics()
            
            # Assertions for concurrent performance
            assert stats['success_rate'] >= 0.90, f"Success rate too low with {num_threads} threads: {stats['success_rate']}"
            assert stats['p95_response_time'] < 10.0, f"P95 response time too high with {num_threads} threads: {stats['p95_response_time']}"
            
            print(f"Threads {num_threads}: {stats['p95_response_time']:.3f}s P95, {stats['throughput']:.1f} req/s")
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        tester = ModelLoadTester()
        
        # Run sustained load test
        results = tester.sustained_load_test(
            duration_seconds=30,
            requests_per_second=10,
            data_shape=(1, 1, 1000)
        )
        
        stats = results.calculate_statistics()
        
        # Memory usage should be reasonable
        assert stats['max_memory_usage'] < 2000, f"Memory usage too high: {stats['max_memory_usage']:.1f} MB"
        
        # Memory should not grow continuously (check for leaks)
        if len(results.memory_usage) > 10:
            early_avg = statistics.mean(results.memory_usage[:10])
            late_avg = statistics.mean(results.memory_usage[-10:])
            memory_growth = late_avg - early_avg
            
            assert memory_growth < 100, f"Potential memory leak detected: {memory_growth:.1f} MB growth"
        
        print(f"Memory usage: {stats['avg_memory_usage']:.1f} MB avg, {stats['max_memory_usage']:.1f} MB max")
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        tester = ModelLoadTester()
        
        # Test large batch processing
        large_batch_data = torch.randn(100, 1, 1000)
        
        start_time = time.time()
        response_time, success, error = tester.single_prediction(large_batch_data)
        end_time = time.time()
        
        assert success, f"Large batch processing failed: {error}"
        
        # Calculate samples per second
        samples_per_second = 100 / response_time
        assert samples_per_second > 20, f"Batch processing too slow: {samples_per_second:.1f} samples/s"
        
        print(f"Batch processing: {samples_per_second:.1f} samples/s")


class TestAPILoadPerformance:
    """Test API performance under load."""
    
    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_prediction_endpoint_load(self):
        """Test prediction endpoint under load."""
        tester = APILoadTester()
        
        def generate_payload():
            """Generate test payload for prediction."""
            return {
                "data": np.random.randn(1000).tolist(),
                "model_type": "cnn"
            }
        
        # Test concurrent requests
        results = tester.concurrent_api_test(
            endpoint="/predict",
            num_requests=100,
            num_threads=10,
            payload_generator=generate_payload
        )
        
        stats = results.calculate_statistics()
        
        # API performance assertions
        assert stats['success_rate'] >= 0.95, f"API success rate too low: {stats['success_rate']}"
        assert stats['avg_response_time'] < 2.0, f"API response time too high: {stats['avg_response_time']}"
        assert stats['throughput'] > 10, f"API throughput too low: {stats['throughput']} req/s"
        
        print(f"API Performance: {stats['avg_response_time']:.3f}s avg, {stats['throughput']:.1f} req/s")
    
    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_batch_prediction_endpoint_load(self):
        """Test batch prediction endpoint under load."""
        tester = APILoadTester()
        
        def generate_batch_payload():
            """Generate test payload for batch prediction."""
            return {
                "data": [np.random.randn(1000).tolist() for _ in range(10)],
                "model_type": "cnn"
            }
        
        # Test batch requests
        results = tester.concurrent_api_test(
            endpoint="/predict/batch",
            num_requests=50,
            num_threads=5,
            payload_generator=generate_batch_payload
        )
        
        stats = results.calculate_statistics()
        
        # Batch API performance assertions
        assert stats['success_rate'] >= 0.90, f"Batch API success rate too low: {stats['success_rate']}"
        assert stats['avg_response_time'] < 10.0, f"Batch API response time too high: {stats['avg_response_time']}"
        
        print(f"Batch API Performance: {stats['avg_response_time']:.3f}s avg, {stats['throughput']:.1f} req/s")


class TestSystemStressTest:
    """System-wide stress testing."""
    
    def test_memory_stress(self):
        """Test system behavior under memory stress."""
        tester = ModelLoadTester()
        
        # Gradually increase load and monitor memory
        memory_usage = []
        
        for batch_size in [1, 10, 50, 100]:
            # Force garbage collection before test
            gc.collect()
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run test with increasing batch size
            results = tester.concurrent_load_test(
                num_requests=20,
                num_threads=2,
                data_shape=(batch_size, 1, 1000)
            )
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase)
            
            stats = results.calculate_statistics()
            
            # System should remain stable
            assert stats['success_rate'] >= 0.80, f"System unstable at batch size {batch_size}"
            
            print(f"Batch size {batch_size}: {memory_increase:.1f} MB increase, {stats['success_rate']:.2f} success rate")
        
        # Memory usage should not grow exponentially
        if len(memory_usage) >= 2:
            growth_rate = memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1
            assert growth_rate < 10, f"Memory usage growing too fast: {growth_rate:.1f}x"
    
    def test_cpu_stress(self):
        """Test system behavior under CPU stress."""
        tester = ModelLoadTester()
        
        # Run high-concurrency test
        results = tester.concurrent_load_test(
            num_requests=500,
            num_threads=20,
            data_shape=(1, 1, 1000)
        )
        
        stats = results.calculate_statistics()
        
        # System should handle high concurrency
        assert stats['success_rate'] >= 0.70, f"System failed under CPU stress: {stats['success_rate']}"
        assert stats['p99_response_time'] < 30.0, f"Response times too high under stress: {stats['p99_response_time']}"
        
        print(f"CPU Stress Test: {stats['success_rate']:.2f} success rate, {stats['p99_response_time']:.3f}s P99")
    
    def test_sustained_load_stability(self):
        """Test system stability under sustained load."""
        tester = ModelLoadTester()
        
        # Run sustained load for extended period
        results = tester.sustained_load_test(
            duration_seconds=60,
            requests_per_second=5,
            data_shape=(1, 1, 1000)
        )
        
        stats = results.calculate_statistics()
        
        # System should remain stable over time
        assert stats['success_rate'] >= 0.85, f"System unstable under sustained load: {stats['success_rate']}"
        
        # Response times should not degrade significantly over time
        if len(results.response_times) > 20:
            early_times = results.response_times[:10]
            late_times = results.response_times[-10:]
            
            early_avg = statistics.mean(early_times)
            late_avg = statistics.mean(late_times)
            
            degradation = (late_avg - early_avg) / early_avg if early_avg > 0 else 0
            assert degradation < 2.0, f"Performance degraded significantly: {degradation:.1f}x slower"
        
        print(f"Sustained Load: {stats['success_rate']:.2f} success rate, {stats['avg_response_time']:.3f}s avg")


class TestResourceLimits:
    """Test system behavior at resource limits."""
    
    def test_maximum_batch_size(self):
        """Test maximum supported batch size."""
        tester = ModelLoadTester()
        
        # Find maximum batch size that works
        max_working_batch = 1
        
        for batch_size in [1, 10, 50, 100, 200, 500, 1000]:
            try:
                results = tester.concurrent_load_test(
                    num_requests=5,
                    num_threads=1,
                    data_shape=(batch_size, 1, 1000)
                )
                
                stats = results.calculate_statistics()
                
                if stats['success_rate'] >= 0.8:
                    max_working_batch = batch_size
                    print(f"Batch size {batch_size}: OK ({stats['avg_response_time']:.3f}s)")
                else:
                    print(f"Batch size {batch_size}: FAILED")
                    break
                    
            except Exception as e:
                print(f"Batch size {batch_size}: ERROR - {e}")
                break
        
        # Should support reasonable batch sizes
        assert max_working_batch >= 10, f"Maximum batch size too small: {max_working_batch}"
        
        print(f"Maximum working batch size: {max_working_batch}")
    
    def test_maximum_concurrent_requests(self):
        """Test maximum concurrent requests supported."""
        tester = ModelLoadTester()
        
        max_working_threads = 1
        
        for num_threads in [1, 2, 5, 10, 20, 50]:
            try:
                results = tester.concurrent_load_test(
                    num_requests=num_threads * 5,
                    num_threads=num_threads,
                    data_shape=(1, 1, 1000)
                )
                
                stats = results.calculate_statistics()
                
                if stats['success_rate'] >= 0.8 and stats['avg_response_time'] < 10.0:
                    max_working_threads = num_threads
                    print(f"Threads {num_threads}: OK ({stats['avg_response_time']:.3f}s, {stats['success_rate']:.2f})")
                else:
                    print(f"Threads {num_threads}: DEGRADED ({stats['avg_response_time']:.3f}s, {stats['success_rate']:.2f})")
                    break
                    
            except Exception as e:
                print(f"Threads {num_threads}: ERROR - {e}")
                break
        
        # Should support reasonable concurrency
        assert max_working_threads >= 2, f"Maximum concurrency too low: {max_working_threads}"
        
        print(f"Maximum working thread count: {max_working_threads}")


if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__,
        "-v",
        "-s",  # Don't capture output
        "--tb=short"
    ])