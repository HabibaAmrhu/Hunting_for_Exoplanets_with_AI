#!/usr/bin/env python3
"""
Production Readiness Validation Script for Exoplanet Detection Pipeline.
Comprehensive validation of all system components before production deployment.
"""

import os
import sys
import json
import time
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import numpy as np
    import pandas as pd
    import requests
    from sklearn.metrics import accuracy_score, f1_score
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


@dataclass
class ValidationResult:
    """Container for validation results."""
    component: str
    test_name: str
    status: str  # PASS, FAIL, SKIP, WARN
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.details is None:
            self.details = {}


class ProductionReadinessValidator:
    """Main validator for production readiness checks."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize validator."""
        self.results: List[ValidationResult] = []
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load validation configuration."""
        default_config = {
            "skip_slow_tests": False,
            "api_timeout": 30,
            "model_accuracy_threshold": 0.7,
            "performance_threshold": {
                "inference_time": 5.0,  # seconds
                "memory_usage": 2048,   # MB
                "cpu_usage": 80         # percent
            },
            "security_checks": True,
            "load_test_duration": 60,
            "required_files": [
                "requirements.txt",
                "setup.py",
                "README.md",
                "LICENSE",
                "src/__init__.py"
            ]
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("production_validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        
        # Log result
        status_color = {
            'PASS': '\033[92m',  # Green
            'FAIL': '\033[91m',  # Red
            'WARN': '\033[93m',  # Yellow
            'SKIP': '\033[94m',  # Blue
        }
        reset_color = '\033[0m'
        
        color = status_color.get(result.status, '')
        print(f"{color}[{result.status}]{reset_color} {result.component}.{result.test_name}: {result.message}")
    
    def validate_dependencies(self) -> bool:
        """Validate all required dependencies are installed."""
        self.logger.info("Validating dependencies...")
        
        if not DEPENDENCIES_AVAILABLE:
            self.add_result(ValidationResult(
                component="dependencies",
                test_name="import_check",
                status="FAIL",
                message=f"Failed to import required dependencies: {IMPORT_ERROR}"
            ))
            return False
        
        # Check specific package versions
        required_packages = {
            'torch': '1.9.0',
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'scikit-learn': '1.0.0'
        }
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                
                self.add_result(ValidationResult(
                    component="dependencies",
                    test_name=f"{package}_version",
                    status="PASS",
                    message=f"{package} version {version} available",
                    details={"version": version, "required": min_version}
                ))
            except ImportError:
                self.add_result(ValidationResult(
                    component="dependencies",
                    test_name=f"{package}_import",
                    status="FAIL",
                    message=f"Required package {package} not available"
                ))
                return False
        
        return True
    
    def validate_project_structure(self) -> bool:
        """Validate project structure and required files."""
        self.logger.info("Validating project structure...")
        
        all_files_exist = True
        
        for required_file in self.config["required_files"]:
            file_path = self.project_root / required_file
            
            if file_path.exists():
                self.add_result(ValidationResult(
                    component="structure",
                    test_name=f"file_{required_file.replace('/', '_')}",
                    status="PASS",
                    message=f"Required file {required_file} exists"
                ))
            else:
                self.add_result(ValidationResult(
                    component="structure",
                    test_name=f"file_{required_file.replace('/', '_')}",
                    status="FAIL",
                    message=f"Required file {required_file} missing"
                ))
                all_files_exist = False
        
        # Check directory structure
        required_dirs = [
            "src",
            "tests",
            "docs",
            "scripts",
            "deployment"
        ]
        
        for required_dir in required_dirs:
            dir_path = self.project_root / required_dir
            
            if dir_path.exists() and dir_path.is_dir():
                self.add_result(ValidationResult(
                    component="structure",
                    test_name=f"dir_{required_dir}",
                    status="PASS",
                    message=f"Required directory {required_dir} exists"
                ))
            else:
                self.add_result(ValidationResult(
                    component="structure",
                    test_name=f"dir_{required_dir}",
                    status="WARN",
                    message=f"Directory {required_dir} missing or not a directory"
                ))
        
        return all_files_exist
    
    def validate_code_quality(self) -> bool:
        """Validate code quality using linting tools."""
        self.logger.info("Validating code quality...")
        
        # Run flake8 linting
        try:
            result = subprocess.run(
                ["flake8", "src/", "--max-line-length=100", "--ignore=E203,W503"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.add_result(ValidationResult(
                    component="code_quality",
                    test_name="flake8_linting",
                    status="PASS",
                    message="Code passes flake8 linting checks"
                ))
            else:
                self.add_result(ValidationResult(
                    component="code_quality",
                    test_name="flake8_linting",
                    status="FAIL",
                    message=f"Linting issues found: {result.stdout[:200]}...",
                    details={"stdout": result.stdout, "stderr": result.stderr}
                ))
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_result(ValidationResult(
                component="code_quality",
                test_name="flake8_linting",
                status="SKIP",
                message="flake8 not available or timed out"
            ))
        
        # Check for TODO/FIXME comments
        todo_count = 0
        fixme_count = 0
        
        for py_file in self.project_root.rglob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    todo_count += content.upper().count('TODO')
                    fixme_count += content.upper().count('FIXME')
            except Exception:
                continue
        
        if todo_count > 0 or fixme_count > 0:
            self.add_result(ValidationResult(
                component="code_quality",
                test_name="todo_fixme_check",
                status="WARN",
                message=f"Found {todo_count} TODO and {fixme_count} FIXME comments",
                details={"todo_count": todo_count, "fixme_count": fixme_count}
            ))
        else:
            self.add_result(ValidationResult(
                component="code_quality",
                test_name="todo_fixme_check",
                status="PASS",
                message="No TODO/FIXME comments found"
            ))
        
        return True
    
    def validate_tests(self) -> bool:
        """Validate test suite execution."""
        self.logger.info("Validating test suite...")
        
        try:
            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--timeout=300"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                # Parse test results
                output_lines = result.stdout.split('\n')
                test_summary = [line for line in output_lines if 'passed' in line and 'failed' in line]
                
                self.add_result(ValidationResult(
                    component="tests",
                    test_name="pytest_execution",
                    status="PASS",
                    message=f"All tests passed: {test_summary[0] if test_summary else 'Tests completed'}",
                    details={"stdout": result.stdout[-500:]}  # Last 500 chars
                ))
                return True
            else:
                self.add_result(ValidationResult(
                    component="tests",
                    test_name="pytest_execution",
                    status="FAIL",
                    message=f"Tests failed with return code {result.returncode}",
                    details={"stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
                ))
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.add_result(ValidationResult(
                component="tests",
                test_name="pytest_execution",
                status="FAIL",
                message=f"Test execution failed: {str(e)}"
            ))
            return False
    
    def validate_model_functionality(self) -> bool:
        """Validate core model functionality."""
        self.logger.info("Validating model functionality...")
        
        try:
            from src.models import ExoplanetCNN
            from src.training import MetricsCalculator
            
            # Create and test model
            model = ExoplanetCNN(input_length=1000)
            
            # Test forward pass
            test_input = torch.randn(10, 1, 1000)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            inference_time = time.time() - start_time
            
            # Validate output
            if output.shape == (10, 1) and torch.all(output >= 0) and torch.all(output <= 1):
                self.add_result(ValidationResult(
                    component="model",
                    test_name="forward_pass",
                    status="PASS",
                    message=f"Model forward pass successful ({inference_time:.3f}s)",
                    details={"inference_time": inference_time, "output_shape": list(output.shape)}
                ))
            else:
                self.add_result(ValidationResult(
                    component="model",
                    test_name="forward_pass",
                    status="FAIL",
                    message=f"Model output validation failed: shape={output.shape}"
                ))
                return False
            
            # Test metrics calculation
            calculator = MetricsCalculator()
            y_true = np.array([0, 1, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 0, 1])
            
            metrics = calculator.calculate_metrics(y_true, y_pred)
            
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            if all(metric in metrics for metric in required_metrics):
                self.add_result(ValidationResult(
                    component="model",
                    test_name="metrics_calculation",
                    status="PASS",
                    message="Metrics calculation successful",
                    details=metrics
                ))
            else:
                self.add_result(ValidationResult(
                    component="model",
                    test_name="metrics_calculation",
                    status="FAIL",
                    message="Missing required metrics"
                ))
                return False
            
            return True
            
        except Exception as e:
            self.add_result(ValidationResult(
                component="model",
                test_name="functionality_check",
                status="FAIL",
                message=f"Model validation failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
            return False
    
    def validate_data_pipeline(self) -> bool:
        """Validate data processing pipeline."""
        self.logger.info("Validating data pipeline...")
        
        try:
            from src.data import ExoplanetDataset, DataAugmentation
            from src.preprocessing import LightCurvePreprocessor
            
            # Test data augmentation
            augmenter = DataAugmentation()
            test_data = np.random.randn(1000)
            
            augmented_data = augmenter.add_noise(test_data, noise_level=0.01)
            
            if augmented_data.shape == test_data.shape:
                self.add_result(ValidationResult(
                    component="data",
                    test_name="augmentation",
                    status="PASS",
                    message="Data augmentation working correctly"
                ))
            else:
                self.add_result(ValidationResult(
                    component="data",
                    test_name="augmentation",
                    status="FAIL",
                    message="Data augmentation shape mismatch"
                ))
                return False
            
            # Test preprocessing
            preprocessor = LightCurvePreprocessor()
            
            # Create synthetic light curve data
            time_data = np.linspace(0, 100, 1000)
            flux_data = np.ones_like(time_data) + 0.01 * np.random.randn(len(time_data))
            
            light_curve = {
                'time': time_data,
                'flux': flux_data,
                'flux_err': 0.01 * np.ones_like(flux_data)
            }
            
            processed_data = preprocessor.process(light_curve)
            
            if isinstance(processed_data, dict) and 'flux' in processed_data:
                self.add_result(ValidationResult(
                    component="data",
                    test_name="preprocessing",
                    status="PASS",
                    message="Data preprocessing working correctly"
                ))
            else:
                self.add_result(ValidationResult(
                    component="data",
                    test_name="preprocessing",
                    status="FAIL",
                    message="Data preprocessing failed"
                ))
                return False
            
            return True
            
        except Exception as e:
            self.add_result(ValidationResult(
                component="data",
                test_name="pipeline_check",
                status="FAIL",
                message=f"Data pipeline validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            return False
    
    def validate_security(self) -> bool:
        """Validate security features."""
        self.logger.info("Validating security features...")
        
        if not self.config["security_checks"]:
            self.add_result(ValidationResult(
                component="security",
                test_name="security_validation",
                status="SKIP",
                message="Security checks disabled in configuration"
            ))
            return True
        
        try:
            from src.security import SecurityManager, UserRole
            
            # Test security manager
            security = SecurityManager()
            
            # Test user creation with strong password
            success, message = security.create_user(
                "testuser", "test@example.com", "StrongPass123!", UserRole.VIEWER
            )
            
            if success:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="user_creation",
                    status="PASS",
                    message="User creation with strong password successful"
                ))
            else:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="user_creation",
                    status="FAIL",
                    message=f"User creation failed: {message}"
                ))
                return False
            
            # Test weak password rejection
            success, message = security.create_user(
                "weakuser", "weak@example.com", "weak", UserRole.VIEWER
            )
            
            if not success:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="weak_password_rejection",
                    status="PASS",
                    message="Weak password correctly rejected"
                ))
            else:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="weak_password_rejection",
                    status="FAIL",
                    message="Weak password was accepted"
                ))
                return False
            
            # Test authentication
            success, user_id, error = security.authenticate_user("testuser", "StrongPass123!")
            
            if success:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="authentication",
                    status="PASS",
                    message="User authentication successful"
                ))
                
                # Test token generation
                token = security.generate_token(user_id)
                is_valid, payload = security.verify_token(token)
                
                if is_valid:
                    self.add_result(ValidationResult(
                        component="security",
                        test_name="token_validation",
                        status="PASS",
                        message="Token generation and validation successful"
                    ))
                else:
                    self.add_result(ValidationResult(
                        component="security",
                        test_name="token_validation",
                        status="FAIL",
                        message="Token validation failed"
                    ))
                    return False
            else:
                self.add_result(ValidationResult(
                    component="security",
                    test_name="authentication",
                    status="FAIL",
                    message=f"Authentication failed: {error}"
                ))
                return False
            
            return True
            
        except Exception as e:
            self.add_result(ValidationResult(
                component="security",
                test_name="security_check",
                status="FAIL",
                message=f"Security validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            return False
    
    def validate_performance(self) -> bool:
        """Validate performance requirements."""
        self.logger.info("Validating performance requirements...")
        
        try:
            from src.models import ExoplanetCNN
            import psutil
            
            # Test inference performance
            model = ExoplanetCNN(input_length=1000)
            model.eval()
            
            # Measure inference time
            test_data = torch.randn(100, 1, 1000)
            
            start_time = time.time()
            with torch.no_grad():
                predictions = model(test_data)
            inference_time = time.time() - start_time
            
            samples_per_second = len(test_data) / inference_time
            
            threshold = self.config["performance_threshold"]["inference_time"]
            
            if inference_time < threshold:
                self.add_result(ValidationResult(
                    component="performance",
                    test_name="inference_speed",
                    status="PASS",
                    message=f"Inference speed: {samples_per_second:.1f} samples/sec",
                    details={"inference_time": inference_time, "samples_per_second": samples_per_second}
                ))
            else:
                self.add_result(ValidationResult(
                    component="performance",
                    test_name="inference_speed",
                    status="FAIL",
                    message=f"Inference too slow: {inference_time:.3f}s > {threshold}s"
                ))
                return False
            
            # Test memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            memory_threshold = self.config["performance_threshold"]["memory_usage"]
            
            if memory_mb < memory_threshold:
                self.add_result(ValidationResult(
                    component="performance",
                    test_name="memory_usage",
                    status="PASS",
                    message=f"Memory usage: {memory_mb:.1f} MB",
                    details={"memory_mb": memory_mb}
                ))
            else:
                self.add_result(ValidationResult(
                    component="performance",
                    test_name="memory_usage",
                    status="WARN",
                    message=f"High memory usage: {memory_mb:.1f} MB > {memory_threshold} MB"
                ))
            
            return True
            
        except Exception as e:
            self.add_result(ValidationResult(
                component="performance",
                test_name="performance_check",
                status="FAIL",
                message=f"Performance validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            return False
    
    def validate_deployment_readiness(self) -> bool:
        """Validate deployment configuration and readiness."""
        self.logger.info("Validating deployment readiness...")
        
        # Check Docker configuration
        dockerfile_path = self.project_root / "deployment" / "docker" / "Dockerfile.prod"
        
        if dockerfile_path.exists():
            self.add_result(ValidationResult(
                component="deployment",
                test_name="dockerfile_exists",
                status="PASS",
                message="Production Dockerfile exists"
            ))
        else:
            self.add_result(ValidationResult(
                component="deployment",
                test_name="dockerfile_exists",
                status="FAIL",
                message="Production Dockerfile missing"
            ))
            return False
        
        # Check Kubernetes configuration
        k8s_path = self.project_root / "deployment" / "kubernetes"
        
        if k8s_path.exists():
            k8s_files = list(k8s_path.glob("*.yaml"))
            if k8s_files:
                self.add_result(ValidationResult(
                    component="deployment",
                    test_name="kubernetes_config",
                    status="PASS",
                    message=f"Kubernetes configuration found: {len(k8s_files)} files"
                ))
            else:
                self.add_result(ValidationResult(
                    component="deployment",
                    test_name="kubernetes_config",
                    status="WARN",
                    message="Kubernetes directory exists but no YAML files found"
                ))
        else:
            self.add_result(ValidationResult(
                component="deployment",
                test_name="kubernetes_config",
                status="WARN",
                message="Kubernetes configuration directory missing"
            ))
        
        # Check environment configuration
        env_files = [
            ".env.example",
            "deployment/config/production.env"
        ]
        
        env_found = False
        for env_file in env_files:
            if (self.project_root / env_file).exists():
                env_found = True
                break
        
        if env_found:
            self.add_result(ValidationResult(
                component="deployment",
                test_name="environment_config",
                status="PASS",
                message="Environment configuration found"
            ))
        else:
            self.add_result(ValidationResult(
                component="deployment",
                test_name="environment_config",
                status="WARN",
                message="No environment configuration files found"
            ))
        
        return True
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        self.logger.info("Validating documentation...")
        
        required_docs = [
            "README.md",
            "docs/api_reference.md",
            "docs/user_guide.md",
            "docs/developer_guide.md",
            "docs/installation.md"
        ]
        
        docs_complete = True
        
        for doc_file in required_docs:
            doc_path = self.project_root / doc_file
            
            if doc_path.exists():
                # Check if file has content
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if len(content) > 100:  # Minimum content check
                        self.add_result(ValidationResult(
                            component="documentation",
                            test_name=f"doc_{doc_file.replace('/', '_').replace('.', '_')}",
                            status="PASS",
                            message=f"Documentation file {doc_file} exists and has content"
                        ))
                    else:
                        self.add_result(ValidationResult(
                            component="documentation",
                            test_name=f"doc_{doc_file.replace('/', '_').replace('.', '_')}",
                            status="WARN",
                            message=f"Documentation file {doc_file} exists but appears incomplete"
                        ))
                        docs_complete = False
                        
                except Exception:
                    self.add_result(ValidationResult(
                        component="documentation",
                        test_name=f"doc_{doc_file.replace('/', '_').replace('.', '_')}",
                        status="FAIL",
                        message=f"Could not read documentation file {doc_file}"
                    ))
                    docs_complete = False
            else:
                self.add_result(ValidationResult(
                    component="documentation",
                    test_name=f"doc_{doc_file.replace('/', '_').replace('.', '_')}",
                    status="FAIL",
                    message=f"Required documentation file {doc_file} missing"
                ))
                docs_complete = False
        
        return docs_complete
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report...")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        warned_tests = len([r for r in self.results if r.status == "WARN"])
        skipped_tests = len([r for r in self.results if r.status == "SKIP"])
        
        # Group results by component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
        
        # Calculate component scores
        component_scores = {}
        for component, results in components.items():
            component_total = len(results)
            component_passed = len([r for r in results if r.status == "PASS"])
            component_scores[component] = component_passed / component_total if component_total > 0 else 0
        
        # Overall readiness score
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        
        # Determine readiness status
        if failed_tests == 0 and overall_score >= 0.9:
            readiness_status = "READY"
        elif failed_tests == 0 and overall_score >= 0.8:
            readiness_status = "MOSTLY_READY"
        elif failed_tests <= 2 and overall_score >= 0.7:
            readiness_status = "NEEDS_ATTENTION"
        else:
            readiness_status = "NOT_READY"
        
        report = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "overall_status": readiness_status,
            "overall_score": overall_score,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warned_tests,
                "skipped": skipped_tests
            },
            "component_scores": component_scores,
            "components": {
                component: [asdict(result) for result in results]
                for component, results in components.items()
            },
            "critical_issues": [
                asdict(result) for result in self.results
                if result.status == "FAIL"
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for critical failures
        failed_results = [r for r in self.results if r.status == "FAIL"]
        
        if failed_results:
            recommendations.append("Address all FAILED tests before production deployment")
            
            # Specific recommendations based on failures
            failed_components = set(r.component for r in failed_results)
            
            if "dependencies" in failed_components:
                recommendations.append("Install missing dependencies: pip install -r requirements.txt")
            
            if "tests" in failed_components:
                recommendations.append("Fix failing tests before deployment")
            
            if "security" in failed_components:
                recommendations.append("Address security issues - these are critical for production")
            
            if "model" in failed_components:
                recommendations.append("Fix model functionality issues")
        
        # Check for warnings
        warned_results = [r for r in self.results if r.status == "WARN"]
        
        if warned_results:
            recommendations.append("Review and address WARNING items for optimal production readiness")
        
        # Performance recommendations
        performance_results = [r for r in self.results if r.component == "performance"]
        if any(r.status in ["FAIL", "WARN"] for r in performance_results):
            recommendations.append("Optimize performance before handling production load")
        
        # Documentation recommendations
        doc_results = [r for r in self.results if r.component == "documentation"]
        if any(r.status == "FAIL" for r in doc_results):
            recommendations.append("Complete missing documentation for production support")
        
        if not recommendations:
            recommendations.append("System appears ready for production deployment!")
        
        return recommendations
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production readiness validation."""
        self.logger.info("Starting production readiness validation...")
        
        start_time = time.time()
        
        # Run all validation checks
        validation_steps = [
            ("Dependencies", self.validate_dependencies),
            ("Project Structure", self.validate_project_structure),
            ("Code Quality", self.validate_code_quality),
            ("Tests", self.validate_tests),
            ("Model Functionality", self.validate_model_functionality),
            ("Data Pipeline", self.validate_data_pipeline),
            ("Security", self.validate_security),
            ("Performance", self.validate_performance),
            ("Deployment Readiness", self.validate_deployment_readiness),
            ("Documentation", self.validate_documentation)
        ]
        
        for step_name, validation_func in validation_steps:
            self.logger.info(f"Running {step_name} validation...")
            try:
                validation_func()
            except Exception as e:
                self.add_result(ValidationResult(
                    component=step_name.lower().replace(" ", "_"),
                    test_name="validation_execution",
                    status="FAIL",
                    message=f"Validation step failed with exception: {str(e)}",
                    details={"error": str(e), "traceback": traceback.format_exc()}
                ))
        
        total_time = time.time() - start_time
        
        # Generate final report
        report = self.generate_report()
        report["validation_duration"] = total_time
        
        self.logger.info(f"Validation completed in {total_time:.2f} seconds")
        self.logger.info(f"Overall status: {report['overall_status']}")
        self.logger.info(f"Score: {report['overall_score']:.2%}")
        
        return report


def main():
    """Main entry point for production readiness validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Readiness Validation")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--output", type=Path, help="Output report file path")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create validator
    validator = ProductionReadinessValidator(config_path=args.config)
    
    if args.verbose:
        validator.logger.setLevel(logging.DEBUG)
    
    # Run validation
    report = validator.run_full_validation()
    
    # Save report
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"production_readiness_report_{timestamp}.json")
    
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    elif args.format == "html":
        # Generate HTML report (simplified)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Production Readiness Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .status-PASS {{ color: green; }}
                .status-FAIL {{ color: red; }}
                .status-WARN {{ color: orange; }}
                .status-SKIP {{ color: blue; }}
                .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Production Readiness Report</h1>
            <div class="summary">
                <h2>Overall Status: {report['overall_status']}</h2>
                <p>Score: {report['overall_score']:.2%}</p>
                <p>Total Tests: {report['summary']['total_tests']}</p>
                <p>Passed: {report['summary']['passed']}</p>
                <p>Failed: {report['summary']['failed']}</p>
                <p>Warnings: {report['summary']['warnings']}</p>
            </div>
            <h2>Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in report['recommendations'])}
            </ul>
        </body>
        </html>
        """
        
        with open(output_path.with_suffix('.html'), 'w') as f:
            f.write(html_content)
    
    print(f"\nReport saved to: {output_path}")
    
    # Exit with appropriate code
    if report['overall_status'] in ['READY', 'MOSTLY_READY']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()