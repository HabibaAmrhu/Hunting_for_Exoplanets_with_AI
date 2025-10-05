"""
Authentication and authorization system for the exoplanet detection API.
Provides JWT-based authentication, role-based access control, and API key management.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import json
from dataclasses import dataclass
from enum import Enum
import secrets
import hashlib
import logging


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    USER = "user"
    GUEST = "guest"


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            role=UserRole(data['role']),
            is_active=data.get('is_active', True),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None
        )


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key_id': self.key_id,
            'key_hash': self.key_hash,
            'name': self.name,
            'user_id': self.user_id,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        """Create from dictionary."""
        return cls(
            key_id=data['key_id'],
            key_hash=data['key_hash'],
            name=data['name'],
            user_id=data['user_id'],
            permissions=data['permissions'],
            is_active=data.get('is_active', True),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None
        )


class AuthenticationError(Exception):
    """Authentication error."""
    pass


class AuthorizationError(Exception):
    """Authorization error."""
    pass


class AuthManager:
    """
    Authentication and authorization manager.
    """
    
    def __init__(
        self,
        secret_key: str,
        redis_client: Optional[redis.Redis] = None,
        token_expiry: int = 3600,
        refresh_token_expiry: int = 86400 * 7  # 7 days
    ):
        """
        Initialize auth manager.
        
        Args:
            secret_key: JWT secret key
            redis_client: Redis client for token storage
            token_expiry: Access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
        """
        self.secret_key = secret_key
        self.redis_client = redis_client
        self.token_expiry = token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        self.algorithm = "HS256"
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for demo (use database in production)
        self.users: Dict[str, User] = {}
        self.user_passwords: Dict[str, str] = {}
        self.api_keys: Dict[str, APIKey] = {}
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for demo."""
        admin_user = User(
            id="admin-001",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN
        )
        
        researcher_user = User(
            id="researcher-001",
            username="researcher",
            email="researcher@example.com",
            role=UserRole.RESEARCHER
        )
        
        # Hash passwords
        admin_password = self._hash_password("admin123")
        researcher_password = self._hash_password("researcher123")
        
        self.users[admin_user.id] = admin_user
        self.users[researcher_user.id] = researcher_user
        self.user_passwords[admin_user.id] = admin_password
        self.user_passwords[researcher_user.id] = researcher_password
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break
        
        if user is None:
            return None
        
        # Verify password
        stored_password = self.user_passwords.get(user.id)
        if stored_password is None or not self._verify_password(password, stored_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        return user
    
    def create_access_token(self, user: User) -> str:
        """
        Create JWT access token.
        
        Args:
            user: User object
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store token in Redis if available
        if self.redis_client:
            self.redis_client.setex(
                f"token:{token}",
                self.token_expiry,
                json.dumps({'user_id': user.id, 'active': True})
            )
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """
        Create JWT refresh token.
        
        Args:
            user: User object
            
        Returns:
            Refresh token string
        """
        payload = {
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(seconds=self.refresh_token_expiry),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis
        if self.redis_client:
            self.redis_client.setex(
                f"refresh_token:{token}",
                self.refresh_token_expiry,
                json.dumps({'user_id': user.id, 'active': True})
            )
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Check if token is blacklisted
            if self.redis_client:
                token_data = self.redis_client.get(f"token:{token}")
                if token_data:
                    token_info = json.loads(token_data)
                    if not token_info.get('active', True):
                        return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get('type') != 'access':
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token string
            
        Returns:
            New access token if successful, None otherwise
        """
        try:
            # Verify refresh token
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get('type') != 'refresh':
                return None
            
            # Check if refresh token is active
            if self.redis_client:
                token_data = self.redis_client.get(f"refresh_token:{refresh_token}")
                if not token_data:
                    return None
                
                token_info = json.loads(token_data)
                if not token_info.get('active', True):
                    return None
            
            # Get user
            user_id = payload['user_id']
            user = self.users.get(user_id)
            
            if user is None or not user.is_active:
                return None
            
            # Create new access token
            return self.create_access_token(user)
            
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str):
        """
        Revoke access token.
        
        Args:
            token: Token to revoke
        """
        if self.redis_client:
            # Mark token as inactive
            self.redis_client.setex(
                f"token:{token}",
                self.token_expiry,
                json.dumps({'active': False})
            )
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> Tuple[str, APIKey]:
        """
        Create API key for user.
        
        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions
            expires_in_days: Expiry in days (None for no expiry)
            
        Returns:
            Tuple of (api_key_string, api_key_object)
        """
        # Generate API key
        key_string = f"exo_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        key_id = secrets.token_urlsafe(16)
        
        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        # Store API key
        self.api_keys[key_id] = api_key
        
        return key_string, api_key
    
    def verify_api_key(self, key_string: str) -> Optional[APIKey]:
        """
        Verify API key.
        
        Args:
            key_string: API key string
            
        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        # Find API key by hash
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash and api_key.is_active:
                if not api_key.is_expired():
                    # Update last used
                    api_key.last_used = datetime.utcnow()
                    return api_key
        
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        return self.users.get(user_id)
    
    def check_permission(self, user: User, required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user: User object
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Define role permissions
        role_permissions = {
            UserRole.RESEARCHER: [
                'read:data',
                'write:data',
                'read:models',
                'write:models',
                'read:predictions',
                'write:predictions'
            ],
            UserRole.USER: [
                'read:data',
                'read:models',
                'read:predictions',
                'write:predictions'
            ],
            UserRole.GUEST: [
                'read:data',
                'read:models',
                'read:predictions'
            ]
        }
        
        user_permissions = role_permissions.get(user.role, [])
        return required_permission in user_permissions


# FastAPI dependencies
security = HTTPBearer()
auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get auth manager instance."""
    global auth_manager
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )
    return auth_manager


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        auth_mgr: Auth manager instance
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    # Verify token
    payload = auth_mgr.verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get user
    user = auth_mgr.get_user_by_id(payload['user_id'])
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


def get_current_user_with_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager)
) -> User:
    """
    Get current user from JWT token or API key.
    
    Args:
        credentials: HTTP authorization credentials
        auth_mgr: Auth manager instance
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    # Try JWT token first
    if token.startswith('eyJ'):  # JWT tokens start with 'eyJ'
        payload = auth_mgr.verify_token(token)
        if payload:
            user = auth_mgr.get_user_by_id(payload['user_id'])
            if user and user.is_active:
                return user
    
    # Try API key
    if token.startswith('exo_'):
        api_key = auth_mgr.verify_api_key(token)
        if api_key:
            user = auth_mgr.get_user_by_id(api_key.user_id)
            if user and user.is_active:
                return user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )


def require_permission(permission: str):
    """
    Decorator to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    def permission_dependency(
        user: User = Depends(get_current_user_with_api_key),
        auth_mgr: AuthManager = Depends(get_auth_manager)
    ) -> User:
        if not auth_mgr.check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return user
    
    return permission_dependency


def require_role(role: UserRole):
    """
    Decorator to require specific role.
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    def role_dependency(
        user: User = Depends(get_current_user_with_api_key)
    ) -> User:
        if user.role != role and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {role.value}"
            )
        return user
    
    return role_dependency


def initialize_auth(secret_key: str, redis_client: Optional[redis.Redis] = None):
    """
    Initialize authentication system.
    
    Args:
        secret_key: JWT secret key
        redis_client: Redis client for token storage
    """
    global auth_manager
    auth_manager = AuthManager(secret_key, redis_client)