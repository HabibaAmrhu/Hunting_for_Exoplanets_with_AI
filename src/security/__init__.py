"""
Security module for the exoplanet detection pipeline.
"""

from .advanced_auth import (
    SecurityManager,
    User,
    UserRole,
    Permission,
    SecurityEvent,
    create_security_manager
)

__all__ = [
    'SecurityManager',
    'User',
    'UserRole',
    'Permission',
    'SecurityEvent',
    'create_security_manager'
]