#!/usr/bin/env python3
"""
Final validation script to verify complete system integration.
Runs comprehensive checks to ensure production readiness.
"""

import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def main():
    """Run final validation checks."""
    print("🚀 Exoplanet Detection Pipeline - Final Validation")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # Change to project directory
    import os
    os.chdir(project_root)
    
    validation_results = []
    
    # 1. Code Quality Checks
    validation_results.append(
        run_command(
            ["python", "-m", "flake8", "src/", "--max-line-length=100"],
            "Code Quality (flake8)"
        )
    )
    
    # 2. Type Checking
    validation_results.append(
        run_command(
            ["python", "-m", "mypy", "src/", "--ignore-missing-imports"],
            "Type Checking (mypy)"
        )
    )
    
    # 3. Security Checks
    validation_results.append(
        run_command(
            ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
            "Security Scan (bandit)"
        )
    )
    
    # 4. Unit Tests
    validation_results.append(
        run_command(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            "Unit Tests (pytest)"
        )
    )
    
    # 5. Integration Tests
    validation_results.append(
        run_command(
            ["python", "tests/test_comprehensive_qa.py"],
            "Integration Tests"
        )
    )
    
    # 6. Performance Tests
    validation_results.append(
        run_command(
            ["python", "tests/test_performance_benchmarks.py"],
            "Performance Benchmarks"
        )
    )
    
    # 7. Security Compliance
    validation_results.append(
        run_command(
            ["python", "tests/test_security_compliance.py"],
            "Security Compliance"
        )
    )
    
    # 8. Production Readiness Check
    validation_results.append(
        run_command(
            ["python", "scripts/production_readiness_check.py"],
            "Production Readiness"
        )
    )
    
    # 9. Docker Build Test
    validation_results.append(
        run_command(
            ["docker", "build", "-f", "deployment/docker/Dockerfile.prod", "-t", "exoplanet-test", "."],
            "Docker Build"
        )
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(validation_results)
    total_tests = len(validation_results)
    
    test_names = [
        "Code Quality",
        "Type Checking", 
        "Security Scan",
        "Unit Tests",
        "Integration Tests",
        "Performance Tests",
        "Security Compliance",
        "Production Readiness",
        "Docker Build"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, validation_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<30} {status}")
    
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("🚀 System is PRODUCTION READY!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} validation(s) failed")
        print("🔧 Please address issues before production deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())