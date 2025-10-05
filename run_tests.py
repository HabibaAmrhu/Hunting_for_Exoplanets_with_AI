#!/usr/bin/env python3
"""
Comprehensive test runner for the exoplanet detection pipeline.
Provides different test suites and reporting options.

Usage:
    python run_tests.py [suite] [options]
    
Test Suites:
    unit        - Run unit tests only (fast)
    integration - Run integration tests only (slower)
    performance - Run performance benchmarks
    gpu         - Run GPU-specific tests
    all         - Run all tests (default)
    
Options:
    --verbose   - Detailed output
    --coverage  - Generate coverage report
    --html      - Generate HTML report
    --parallel  - Run tests in parallel
    --quick     - Skip slow tests
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def print_info(message):
    """Print info message."""
    print(f"ℹ️  {message}")

class TestRunner:
    """Comprehensive test runner for the pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / 'tests'
        self.src_dir = self.project_root / 'src'
        
    def check_dependencies(self):
        """Check if required testing dependencies are available."""
        required_packages = ['pytest']
        optional_packages = ['pytest-cov', 'pytest-xdist', 'pytest-html']
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_required.append(package)
        
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_optional.append(package)
        
        if missing_required:
            print_error(f"Missing required packages: {missing_required}")
            print_info("Install with: pip install " + " ".join(missing_required))
            return False
        
        if missing_optional:
            print_info(f"Optional packages not available: {missing_optional}")
            print_info("Install with: pip install " + " ".join(missing_optional))
        
        return True
    
    def run_unit_tests(self, verbose=False, coverage=False, html=False, parallel=False):
        """Run unit tests."""
        print_header("Unit Tests")
        
        cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'unit']
        
        if verbose:
            cmd.append('-v')
        
        if coverage:
            cmd.extend(['--cov=src', '--cov-report=term-missing'])
            if html:
                cmd.append('--cov-report=html')
        
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(['-n', 'auto'])
            except ImportError:
                print_info("pytest-xdist not available, running sequentially")
        
        if html:
            try:
                import pytest_html
                cmd.extend(['--html=reports/unit_tests.html', '--self-contained-html'])
            except ImportError:
                print_info("pytest-html not available, skipping HTML report")
        
        return self._run_command(cmd)
    
    def run_integration_tests(self, verbose=False, quick=False):
        """Run integration tests."""
        print_header("Integration Tests")
        
        cmd = ['python', '-m', 'pytest', 'tests/test_integration.py', '-m', 'integration']
        
        if verbose:
            cmd.append('-v')
        
        if quick:
            cmd.extend(['-m', 'not slow'])
        
        return self._run_command(cmd)
    
    def run_performance_tests(self, verbose=False):
        """Run performance benchmark tests."""
        print_header("Performance Tests")
        
        cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'performance']
        
        if verbose:
            cmd.append('-v')
        
        cmd.append('-s')  # Don't capture output for performance tests
        
        return self._run_command(cmd)
    
    def run_gpu_tests(self, verbose=False):
        """Run GPU-specific tests."""
        print_header("GPU Tests")
        
        # Check if CUDA is available
        try:
            import torch
            if not torch.cuda.is_available():
                print_info("CUDA not available, skipping GPU tests")
                return True
        except ImportError:
            print_error("PyTorch not available, cannot run GPU tests")
            return False
        
        cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'gpu']
        
        if verbose:
            cmd.append('-v')
        
        return self._run_command(cmd)
    
    def run_all_tests(self, verbose=False, coverage=False, html=False, parallel=False, quick=False):
        """Run all tests."""
        print_header("Complete Test Suite")
        
        cmd = ['python', '-m', 'pytest', 'tests/']
        
        if verbose:
            cmd.append('-v')
        
        if quick:
            cmd.extend(['-m', 'not slow'])
        
        if coverage:
            cmd.extend(['--cov=src', '--cov-report=term-missing'])
            if html:
                cmd.append('--cov-report=html')
        
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(['-n', 'auto'])
            except ImportError:
                print_info("pytest-xdist not available, running sequentially")
        
        if html:
            try:
                import pytest_html
                cmd.extend(['--html=reports/all_tests.html', '--self-contained-html'])
            except ImportError:
                print_info("pytest-html not available, skipping HTML report")
        
        return self._run_command(cmd)
    
    def run_quick_validation(self):
        """Run quick validation script."""
        print_header("Quick Validation")
        
        cmd = ['python', 'run_quick_test.py', '--verbose']
        return self._run_command(cmd)
    
    def _run_command(self, cmd):
        """Run a command and return success status."""
        try:
            print_info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Command failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print_error("Command not found. Make sure pytest is installed.")
            return False
    
    def create_reports_dir(self):
        """Create reports directory if it doesn't exist."""
        reports_dir = self.project_root / 'reports'
        reports_dir.mkdir(exist_ok=True)
        return reports_dir
    
    def run_test_suite(self, suite, **kwargs):
        """Run specified test suite."""
        self.create_reports_dir()
        
        start_time = time.time()
        
        if suite == 'unit':
            success = self.run_unit_tests(**kwargs)
        elif suite == 'integration':
            success = self.run_integration_tests(**kwargs)
        elif suite == 'performance':
            success = self.run_performance_tests(**kwargs)
        elif suite == 'gpu':
            success = self.run_gpu_tests(**kwargs)
        elif suite == 'all':
            success = self.run_all_tests(**kwargs)
        elif suite == 'quick':
            success = self.run_quick_validation()
        else:
            print_error(f"Unknown test suite: {suite}")
            return False
        
        end_time = time.time()
        duration = end_time - start_time
        
        print_header("Test Summary")
        
        if success:
            print_success(f"Test suite '{suite}' completed successfully!")
        else:
            print_error(f"Test suite '{suite}' failed!")
        
        print_info(f"Total time: {duration:.2f} seconds")
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for exoplanet detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
    unit        - Unit tests only (fast, ~1-2 minutes)
    integration - Integration tests (slower, ~5-10 minutes)
    performance - Performance benchmarks (~2-5 minutes)
    gpu         - GPU-specific tests (requires CUDA)
    all         - Complete test suite (~10-20 minutes)
    quick       - Quick validation only (~30 seconds)

Examples:
    python run_tests.py unit --verbose
    python run_tests.py all --coverage --html
    python run_tests.py integration --quick
    python run_tests.py performance
        """
    )
    
    parser.add_argument('suite', nargs='?', default='all',
                       choices=['unit', 'integration', 'performance', 'gpu', 'all', 'quick'],
                       help='Test suite to run (default: all)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML reports')
    
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel (requires pytest-xdist)')
    
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)
    
    # Run tests
    success = runner.run_test_suite(
        args.suite,
        verbose=args.verbose,
        coverage=args.coverage,
        html=args.html,
        parallel=args.parallel,
        quick=args.quick
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()