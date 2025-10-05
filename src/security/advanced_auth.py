"""
Advanced security and authentication system for the exoplanet detection pipeline.
Provides JWT authentication, role-based access control, and security monitoring.
"""

import jwt
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import time
from functools import wraps
import re

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(Enum):
    """System permissions."""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    MANAGE_USERS = "manage_users"
    VIEW_METRICS = "view_metrics"
    EXPORT_RESULTS = "export_results"


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: str = "INFO"


class SecurityManager:
    """
    Advanced security manager with authentication and monitoring.
    
    Provides JWT-based authentication, role-based access control,
    password security, and security event monitoring.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry_hours: int = 24,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30
    ):
        """
        Initialize security manager.
        
        Args:
            secret_key: JWT secret key
            token_expiry_hours: Token expiration time
            max_failed_attempts: Maximum failed login attempts
            lockout_duration_minutes: Account lockout duration
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.token_expiry_hours = token_expiry_hours
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.security_events: List[SecurityEvent] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: list(Permission),
            UserRole.RESEARCHER: [
                Permission.READ_DATA, Permission.WRITE_DATA,
                Permission.TRAIN_MODELS, Permission.VIEW_METRICS,
                Permission.EXPORT_RESULTS
            ],
            UserRole.ANALYST: [
                Permission.READ_DATA, Permission.VIEW_METRICS,
                Permission.EXPORT_RESULTS
            ],
            UserRole.VIEWER: [
                Permission.READ_DATA, Permission.VIEW_METRICS
            ]
        }
        
        # Initialize encryption if available
        if CRYPTOGRAPHY_AVAILABLE:
            self.cipher_key = Fernet.generate_key()
            self.cipher = Fernet(self.cipher_key)
        else:
            self.cipher = None
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            password_hash: Stored password hash
            
        Returns:
            True if password matches
        """
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole,
        admin_user_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role
            admin_user_id: ID of admin creating the user
            
        Returns:
            Tuple of (success, message)
        """
        # Validate admin permissions
        if admin_user_id and not self.has_permission(admin_user_id, Permission.MANAGE_USERS):
            return False, "Insufficient permissions to create users"
        
        # Check if user already exists
        if any(u.username == username or u.email == email for u in self.users.values()):
            return False, "User with this username or email already exists"
        
        # Validate password strength
        is_valid, errors = self.validate_password_strength(password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Create user
        user_id = self._generate_user_id()
        password_hash = self.hash_password(password)
        permissions = self.role_permissions.get(role, [])
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        
        # Log security event
        self._log_security_event(
            "USER_CREATED",
            admin_user_id,
            "127.0.0.1",
            {"new_user_id": user_id, "username": username, "role": role.value}
        )
        
        return True, f"User {username} created successfully"
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str = "127.0.0.1"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Authenticate user credentials.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            
        Returns:
            Tuple of (success, user_id, error_message)
        """
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            self._log_security_event(
                "LOGIN_FAILED",
                None,
                ip_address,
                {"username": username, "reason": "user_not_found"}
            )
            return False, None, "Invalid credentials"
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            self._log_security_event(
                "LOGIN_BLOCKED",
                user.user_id,
                ip_address,
                {"reason": "account_locked"}
            )
            return False, None, "Account is temporarily locked"
        
        # Check if account is active
        if not user.is_active:
            self._log_security_event(
                "LOGIN_BLOCKED",
                user.user_id,
                ip_address,
                {"reason": "account_inactive"}
            )
            return False, None, "Account is inactive"
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                self._log_security_event(
                    "ACCOUNT_LOCKED",
                    user.user_id,
                    ip_address,
                    {"failed_attempts": user.failed_login_attempts}
                )
            
            self._log_security_event(
                "LOGIN_FAILED",
                user.user_id,
                ip_address,
                {"reason": "invalid_password", "attempts": user.failed_login_attempts}
            )
            return False, None, "Invalid credentials"
        
        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        self._log_security_event(
            "LOGIN_SUCCESS",
            user.user_id,
            ip_address,
            {"username": user.username}
        )
        
        return True, user.user_id, None
    
    def generate_token(self, user_id: str) -> str:
        """
        Generate JWT token for user.
        
        Args:
            user_id: User ID
            
        Returns:
            JWT token
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        payload = {
            'user_id': user_id,
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store active session
        session_id = hashlib.sha256(token.encode()).hexdigest()
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow()
        }
        
        return token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            session_id = hashlib.sha256(token.encode()).hexdigest()
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['last_activity'] = datetime.utcnow()
            
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    def require_permission(self, permission: Permission):
        """
        Decorator to require specific permission.
        
        Args:
            permission: Required permission
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user_id from function arguments or context
                user_id = kwargs.get('user_id') or (args[0] if args else None)
                
                if not user_id or not self.has_permission(user_id, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke JWT token.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if token was revoked
        """
        session_id = hashlib.sha256(token.encode()).hexdigest()
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data['last_activity']
            if current_time - last_activity > timedelta(hours=self.token_expiry_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
    
    def get_security_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """
        Get security events.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of security events
        """
        events = self.security_events
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        return f"user_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            timestamp=datetime.utcnow(),
            details=details,
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Log to file
        self.logger.info(
            f"Security Event: {event_type} - User: {user_id} - IP: {ip_address} - Details: {details}"
        )
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.cipher:
            raise RuntimeError("Encryption not available")
        
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.cipher:
            raise RuntimeError("Encryption not available")
        
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def export_security_report(self, output_path: Path) -> Dict[str, Any]:
        """
        Export security report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Security report data
        """
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_users': len(self.users),
            'active_users': len([u for u in self.users.values() if u.is_active]),
            'locked_users': len([u for u in self.users.values() if u.locked_until]),
            'active_sessions': len(self.active_sessions),
            'security_events': {
                'total': len(self.security_events),
                'by_type': {},
                'recent_events': []
            }
        }
        
        # Count events by type
        for event in self.security_events:
            event_type = event.event_type
            if event_type not in report['security_events']['by_type']:
                report['security_events']['by_type'][event_type] = 0
            report['security_events']['by_type'][event_type] += 1
        
        # Recent events (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_events = [
            {
                'event_type': e.event_type,
                'user_id': e.user_id,
                'timestamp': e.timestamp.isoformat(),
                'severity': e.severity
            }
            for e in self.security_events
            if e.timestamp > recent_cutoff
        ]
        report['security_events']['recent_events'] = recent_events
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def create_security_manager(
    secret_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> SecurityManager:
    """
    Factory function to create security manager.
    
    Args:
        secret_key: JWT secret key
        config: Configuration dictionary
        
    Returns:
        Configured security manager
    """
    if config is None:
        config = {}
    
    return SecurityManager(
        secret_key=secret_key,
        token_expiry_hours=config.get('token_expiry_hours', 24),
        max_failed_attempts=config.get('max_failed_attempts', 5),
        lockout_duration_minutes=config.get('lockout_duration_minutes', 30)
    )


# Example usage and testing
if __name__ == "__main__":
    # Create security manager
    security = create_security_manager()
    
    # Create admin user
    success, message = security.create_user(
        "admin",
        "admin@example.com",
        "AdminPass123!",
        UserRole.ADMIN
    )
    print(f"Admin creation: {success} - {message}")
    
    # Authenticate user
    success, user_id, error = security.authenticate_user("admin", "AdminPass123!")
    print(f"Authentication: {success} - User ID: {user_id}")
    
    if success:
        # Generate token
        token = security.generate_token(user_id)
        print(f"Token generated: {token[:50]}...")
        
        # Verify token
        is_valid, payload = security.verify_token(token)
        print(f"Token valid: {is_valid} - Payload: {payload}")
        
        # Check permissions
        has_perm = security.has_permission(user_id, Permission.MANAGE_USERS)
        print(f"Has manage users permission: {has_perm}")