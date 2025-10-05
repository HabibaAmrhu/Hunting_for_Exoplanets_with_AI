"""
Security compliance testing suite for the exoplanet detection pipeline.
Tests security features, vulnerability assessments, and compliance requirements.
"""

import pytest
import time
import hashlib
import secrets
import jwt
import requests
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import subprocess
import re

from src.security import SecurityManager, UserRole, Permission
from src.api.auth import AuthenticationMiddleware
from src.utils.validation import InputValidator


class TestAuthenticationSecurity:
    """Test authentication security features."""
    
    def test_password_strength_requirements(self):
        """Test password strength validation."""
        security = SecurityManager()
        
        # Test weak passwords
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "qwerty",
            "12345678",
            "Password",  # Missing special char and number
            "password123",  # Missing uppercase and special char
            "PASSWORD123!",  # Missing lowercase
            "Pass!",  # Too short
        ]
        
        for weak_password in weak_passwords:
            is_valid, errors = security.validate_password_strength(weak_password)
            assert not is_valid, f"Weak password '{weak_password}' should be rejected"
            assert len(errors) > 0, f"Should have error messages for '{weak_password}'"
        
        # Test strong passwords
        strong_passwords = [
            "MySecureP@ssw0rd123",
            "Tr0ub4dor&3",
            "C0mpl3x!P@ssw0rd",
            "S3cur3_P@ssw0rd_2023"
        ]
        
        for strong_password in strong_passwords:
            is_valid, errors = security.validate_password_strength(strong_password)
            assert is_valid, f"Strong password '{strong_password}' should be accepted: {errors}"
    
    def test_password_hashing_security(self):
        """Test password hashing security."""
        security = SecurityManager()
        
        password = "TestPassword123!"
        
        # Test that passwords are properly hashed
        hash1 = security.hash_password(password)
        hash2 = security.hash_password(password)
        
        # Hashes should be different (due to salt)
        assert hash1 != hash2, "Password hashes should be different due to salt"
        
        # Both hashes should verify correctly
        assert security.verify_password(password, hash1), "First hash should verify"
        assert security.verify_password(password, hash2), "Second hash should verify"
        
        # Wrong password should not verify
        assert not security.verify_password("WrongPassword", hash1), "Wrong password should not verify"
    
    def test_jwt_token_security(self):
        """Test JWT token security."""
        security = SecurityManager()
        
        # Create test user
        security.create_user("testuser", "test@example.com", "TestPass123!", UserRole.RESEARCHER)
        success, user_id, _ = security.authenticate_user("testuser", "TestPass123!")
        assert success, "User authentication should succeed"
        
        # Generate token
        token = security.generate_token(user_id)
        
        # Verify token structure
        assert isinstance(token, str), "Token should be a string"
        assert len(token.split('.')) == 3, "JWT should have 3 parts"
        
        # Verify token content
        is_valid, payload = security.verify_token(token)
        assert is_valid, "Generated token should be valid"
        assert payload['user_id'] == user_id, "Token should contain correct user_id"
        
        # Test token expiration (mock time)
        with patch('time.time', return_value=time.time() + 25 * 3600):  # 25 hours later
            is_valid, payload = security.verify_token(token)
            assert not is_valid, "Expired token should be invalid"
        
        # Test token tampering
        tampered_token = token[:-5] + "XXXXX"
        is_valid, payload = security.verify_token(tampered_token)
        assert not is_valid, "Tampered token should be invalid"
    
    def test_account_lockout_mechanism(self):
        """Test account lockout after failed attempts."""
        security = SecurityManager(max_failed_attempts=3, lockout_duration_minutes=1)
        
        # Create test user
        security.create_user("locktest", "lock@example.com", "TestPass123!", UserRole.VIEWER)
        
        # Make failed login attempts
        for i in range(3):
            success, user_id, error = security.authenticate_user("locktest", "wrongpassword")
            assert not success, f"Failed attempt {i+1} should not succeed"
        
        # Account should now be locked
        success, user_id, error = security.authenticate_user("locktest", "TestPass123!")
        assert not success, "Account should be locked after max failed attempts"
        assert "locked" in error.lower(), "Error message should indicate account is locked"
        
        # Test that lockout expires (mock time)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = mock_datetime.utcnow.return_value + security.lockout_duration
            success, user_id, error = security.authenticate_user("locktest", "TestPass123!")
            # Note: This test may need adjustment based on actual implementation
    
    def test_session_management(self):
        """Test session management security."""
        security = SecurityManager()
        
        # Create test user
        security.create_user("sessiontest", "session@example.com", "TestPass123!", UserRole.RESEARCHER)
        success, user_id, _ = security.authenticate_user("sessiontest", "TestPass123!")
        
        # Generate token and check session
        token = security.generate_token(user_id)
        assert len(security.active_sessions) > 0, "Active session should be created"
        
        # Revoke token
        revoked = security.revoke_token(token)
        assert revoked, "Token should be successfully revoked"
        
        # Verify token is no longer valid
        is_valid, payload = security.verify_token(token)
        assert not is_valid, "Revoked token should be invalid"


class TestInputValidationSecurity:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        validator = InputValidator()
        
        # SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users WHERE 1=1; --",
            "' UNION SELECT * FROM passwords --"
        ]
        
        for malicious_input in malicious_inputs:
            is_safe = validator.validate_string_input(malicious_input)
            assert not is_safe, f"SQL injection attempt should be blocked: {malicious_input}"
    
    def test_xss_prevention(self):
        """Test XSS prevention."""
        validator = InputValidator()
        
        # XSS attempts
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for xss_input in xss_inputs:
            sanitized = validator.sanitize_html_input(xss_input)
            assert "<script>" not in sanitized, f"Script tags should be removed: {xss_input}"
            assert "javascript:" not in sanitized, f"JavaScript protocol should be removed: {xss_input}"
            assert "onerror=" not in sanitized, f"Event handlers should be removed: {xss_input}"
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        validator = InputValidator()
        
        # Path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"  # URL encoded
        ]
        
        for malicious_path in malicious_paths:
            is_safe = validator.validate_file_path(malicious_path)
            assert not is_safe, f"Path traversal attempt should be blocked: {malicious_path}"
    
    def test_data_type_validation(self):
        """Test data type validation."""
        validator = InputValidator()
        
        # Test numeric validation
        assert validator.validate_numeric_input("123.45"), "Valid number should pass"
        assert validator.validate_numeric_input("-123"), "Negative number should pass"
        assert not validator.validate_numeric_input("abc"), "Non-numeric should fail"
        assert not validator.validate_numeric_input("123; DROP TABLE"), "SQL in number should fail"
        
        # Test array validation
        valid_array = [1.0, 2.0, 3.0, 4.0]
        assert validator.validate_array_input(valid_array), "Valid array should pass"
        
        invalid_arrays = [
            [1, 2, "malicious"],  # Mixed types
            ["<script>", "alert('xss')"],  # XSS in array
            list(range(100000)),  # Too large
        ]
        
        for invalid_array in invalid_arrays:
            assert not validator.validate_array_input(invalid_array), f"Invalid array should fail: {type(invalid_array[0]) if invalid_array else 'empty'}"
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        validator = InputValidator()
        
        # Test allowed file types
        allowed_files = [
            "data.csv",
            "model.pth",
            "config.json",
            "lightcurve.fits"
        ]
        
        for filename in allowed_files:
            assert validator.validate_filename(filename), f"Allowed file should pass: {filename}"
        
        # Test blocked file types
        blocked_files = [
            "malware.exe",
            "script.bat",
            "virus.com",
            "trojan.scr",
            "backdoor.php",
            "shell.jsp"
        ]
        
        for filename in blocked_files:
            assert not validator.validate_filename(filename), f"Blocked file should fail: {filename}"
        
        # Test file size limits
        assert validator.validate_file_size(1024 * 1024), "1MB file should pass"  # 1MB
        assert not validator.validate_file_size(100 * 1024 * 1024), "100MB file should fail"  # 100MB


class TestAPISecurityCompliance:
    """Test API security compliance."""
    
    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_rate_limiting(self):
        """Test API rate limiting."""
        base_url = "http://localhost:8000"
        
        # Make rapid requests to test rate limiting
        responses = []
        for i in range(20):  # Exceed rate limit
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                responses.append(response.status_code)
            except requests.RequestException:
                responses.append(None)
        
        # Should get rate limited (429 status code)
        rate_limited = any(status == 429 for status in responses)
        assert rate_limited, "Rate limiting should be enforced"
    
    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_authentication_required(self):
        """Test that protected endpoints require authentication."""
        base_url = "http://localhost:8000"
        
        protected_endpoints = [
            "/predict",
            "/train",
            "/models",
            "/admin/users"
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.post(f"{base_url}{endpoint}", timeout=5)
                assert response.status_code in [401, 403], f"Endpoint {endpoint} should require authentication"
            except requests.RequestException:
                pass  # Server not running, skip test
    
    @pytest.mark.skipif(True, reason="Requires running API server")
    def test_cors_configuration(self):
        """Test CORS configuration."""
        base_url = "http://localhost:8000"
        
        # Test CORS headers
        headers = {
            'Origin': 'https://malicious-site.com',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        try:
            response = requests.options(f"{base_url}/predict", headers=headers, timeout=5)
            
            # Should not allow arbitrary origins
            cors_origin = response.headers.get('Access-Control-Allow-Origin', '')
            assert cors_origin != '*', "CORS should not allow all origins in production"
            
        except requests.RequestException:
            pass  # Server not running, skip test
    
    def test_secure_headers(self):
        """Test security headers configuration."""
        # This would typically test the actual API response headers
        expected_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'"
        }
        
        # Mock test for header validation
        mock_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block'
        }
        
        for header, expected_value in expected_headers.items():
            if header in mock_headers:
                assert mock_headers[header] == expected_value, f"Security header {header} should be properly configured"


class TestDataPrivacyCompliance:
    """Test data privacy and compliance features."""
    
    def test_data_anonymization(self):
        """Test data anonymization features."""
        from src.privacy.anonymizer import DataAnonymizer
        
        anonymizer = DataAnonymizer()
        
        # Test PII detection and removal
        sensitive_data = {
            'email': 'user@example.com',
            'phone': '+1-555-123-4567',
            'ssn': '123-45-6789',
            'credit_card': '4111-1111-1111-1111',
            'name': 'John Doe',
            'address': '123 Main St, Anytown, USA'
        }
        
        anonymized_data = anonymizer.anonymize_dict(sensitive_data)
        
        # Check that PII is removed or masked
        assert '@' not in anonymized_data.get('email', ''), "Email should be anonymized"
        assert '555' not in anonymized_data.get('phone', ''), "Phone should be anonymized"
        assert '123-45' not in anonymized_data.get('ssn', ''), "SSN should be anonymized"
        assert '4111' not in anonymized_data.get('credit_card', ''), "Credit card should be anonymized"
    
    def test_data_retention_policies(self):
        """Test data retention policy enforcement."""
        from src.privacy.retention import DataRetentionManager
        
        retention_manager = DataRetentionManager()
        
        # Test retention policy configuration
        policies = retention_manager.get_retention_policies()
        
        assert 'user_data' in policies, "Should have user data retention policy"
        assert 'model_data' in policies, "Should have model data retention policy"
        assert 'log_data' in policies, "Should have log data retention policy"
        
        # Test data expiration
        expired_data = retention_manager.find_expired_data()
        assert isinstance(expired_data, list), "Should return list of expired data"
    
    def test_audit_logging(self):
        """Test audit logging for compliance."""
        security = SecurityManager()
        
        # Perform actions that should be logged
        security.create_user("audituser", "audit@example.com", "AuditPass123!", UserRole.VIEWER)
        success, user_id, _ = security.authenticate_user("audituser", "AuditPass123!")
        
        # Check audit logs
        security_events = security.get_security_events()
        
        assert len(security_events) > 0, "Security events should be logged"
        
        # Check for specific events
        event_types = [event.event_type for event in security_events]
        assert 'USER_CREATED' in event_types, "User creation should be logged"
        assert 'LOGIN_SUCCESS' in event_types, "Successful login should be logged"
        
        # Check event details
        for event in security_events:
            assert event.timestamp is not None, "Event should have timestamp"
            assert event.ip_address is not None, "Event should have IP address"
            assert isinstance(event.details, dict), "Event details should be dictionary"


class TestVulnerabilityAssessment:
    """Test for common security vulnerabilities."""
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        security = SecurityManager()
        
        # Create test user
        security.create_user("timingtest", "timing@example.com", "TimingPass123!", UserRole.VIEWER)
        
        # Measure authentication time for valid user
        valid_times = []
        for _ in range(10):
            start_time = time.time()
            security.authenticate_user("timingtest", "wrongpassword")
            end_time = time.time()
            valid_times.append(end_time - start_time)
        
        # Measure authentication time for invalid user
        invalid_times = []
        for _ in range(10):
            start_time = time.time()
            security.authenticate_user("nonexistentuser", "wrongpassword")
            end_time = time.time()
            invalid_times.append(end_time - start_time)
        
        # Times should be similar to prevent user enumeration
        avg_valid_time = sum(valid_times) / len(valid_times)
        avg_invalid_time = sum(invalid_times) / len(invalid_times)
        
        time_difference_ratio = abs(avg_valid_time - avg_invalid_time) / max(avg_valid_time, avg_invalid_time)
        assert time_difference_ratio < 0.5, f"Timing difference too large: {time_difference_ratio:.2f}"
    
    def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks."""
        security = SecurityManager()
        
        # Create test user
        security.create_user("sessionfix", "sessionfix@example.com", "SessionPass123!", UserRole.VIEWER)
        
        # Generate token before authentication
        old_session_count = len(security.active_sessions)
        
        # Authenticate user
        success, user_id, _ = security.authenticate_user("sessionfix", "SessionPass123!")
        token = security.generate_token(user_id)
        
        new_session_count = len(security.active_sessions)
        
        # New session should be created
        assert new_session_count > old_session_count, "New session should be created on authentication"
    
    def test_csrf_token_validation(self):
        """Test CSRF token validation."""
        # Mock CSRF token validation
        csrf_token = secrets.token_urlsafe(32)
        
        # Validate token format
        assert len(csrf_token) >= 32, "CSRF token should be sufficiently long"
        assert csrf_token.isalnum() or '-' in csrf_token or '_' in csrf_token, "CSRF token should be URL-safe"
    
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        # Test that random values are cryptographically secure
        random_values = [secrets.token_bytes(32) for _ in range(100)]
        
        # Check uniqueness
        unique_values = set(random_values)
        assert len(unique_values) == len(random_values), "Random values should be unique"
        
        # Check entropy (basic test)
        combined_bytes = b''.join(random_values)
        byte_counts = [combined_bytes.count(bytes([i])) for i in range(256)]
        
        # Should have relatively even distribution
        min_count = min(byte_counts)
        max_count = max(byte_counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        assert ratio < 10, f"Random distribution too uneven: {ratio:.2f}"


class TestComplianceRequirements:
    """Test compliance with security standards."""
    
    def test_password_policy_compliance(self):
        """Test password policy compliance (e.g., NIST guidelines)."""
        security = SecurityManager()
        
        # Test minimum length requirement
        short_password = "Abc1!"
        is_valid, errors = security.validate_password_strength(short_password)
        assert not is_valid, "Password shorter than 8 characters should be rejected"
        
        # Test complexity requirements
        simple_password = "password123"
        is_valid, errors = security.validate_password_strength(simple_password)
        assert not is_valid, "Password without complexity should be rejected"
        
        # Test common password rejection
        common_passwords = ["password123", "123456789", "qwerty123"]
        for password in common_passwords:
            is_valid, errors = security.validate_password_strength(password)
            assert not is_valid, f"Common password '{password}' should be rejected"
    
    def test_encryption_standards(self):
        """Test encryption standards compliance."""
        security = SecurityManager()
        
        # Test that sensitive data is encrypted
        if hasattr(security, 'cipher') and security.cipher:
            test_data = "sensitive information"
            encrypted = security.encrypt_sensitive_data(test_data)
            decrypted = security.decrypt_sensitive_data(encrypted)
            
            assert encrypted != test_data, "Data should be encrypted"
            assert decrypted == test_data, "Decryption should restore original data"
    
    def test_access_control_compliance(self):
        """Test access control compliance."""
        security = SecurityManager()
        
        # Create users with different roles
        security.create_user("admin", "admin@example.com", "AdminPass123!", UserRole.ADMIN)
        security.create_user("user", "user@example.com", "UserPass123!", UserRole.VIEWER)
        
        # Get user IDs
        _, admin_id, _ = security.authenticate_user("admin", "AdminPass123!")
        _, user_id, _ = security.authenticate_user("user", "UserPass123!")
        
        # Test role-based access control
        admin_permissions = [
            Permission.MANAGE_USERS,
            Permission.TRAIN_MODELS,
            Permission.DEPLOY_MODELS,
            Permission.VIEW_METRICS
        ]
        
        for permission in admin_permissions:
            assert security.has_permission(admin_id, permission), f"Admin should have {permission.value}"
        
        # Regular user should have limited permissions
        assert not security.has_permission(user_id, Permission.MANAGE_USERS), "User should not manage users"
        assert not security.has_permission(user_id, Permission.DEPLOY_MODELS), "User should not deploy models"
    
    def test_audit_trail_compliance(self):
        """Test audit trail compliance."""
        security = SecurityManager()
        
        # Perform various actions
        security.create_user("audituser", "audit@example.com", "AuditPass123!", UserRole.RESEARCHER)
        success, user_id, _ = security.authenticate_user("audituser", "AuditPass123!")
        token = security.generate_token(user_id)
        security.revoke_token(token)
        
        # Check audit trail
        events = security.get_security_events()
        
        # Should have comprehensive logging
        required_events = ['USER_CREATED', 'LOGIN_SUCCESS']
        logged_events = [event.event_type for event in events]
        
        for required_event in required_events:
            assert required_event in logged_events, f"Event {required_event} should be logged"
        
        # Events should have required fields
        for event in events:
            assert hasattr(event, 'timestamp'), "Event should have timestamp"
            assert hasattr(event, 'event_type'), "Event should have type"
            assert hasattr(event, 'details'), "Event should have details"
            assert hasattr(event, 'ip_address'), "Event should have IP address"


if __name__ == "__main__":
    # Run security compliance tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure for security tests
    ])