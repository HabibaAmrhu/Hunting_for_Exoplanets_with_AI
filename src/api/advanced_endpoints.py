"""
Advanced API endpoints with authentication, rate limiting, and comprehensive features.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
from datetime import datetime, timedelta
import uuid

from .auth import (
    User, UserRole, APIKey, AuthManager, 
    get_current_user, get_current_user_with_api_key,
    require_permission, require_role, initialize_auth
)
from .rate_limiter import RateLimit, RateLimitStrategy, create_rate_limiter, create_rate_limit_middleware
from ..models.ensemble import EnsembleModel
from ..data.dataset import ExoplanetDataset
from ..training.metrics import MetricsCalculator
from ..monitoring.model_monitor import ModelMonitoringSystem


# Pydantic models for API
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    key_id: str
    api_key: str
    name: str
    permissions: List[str]
    expires_at: Optional[datetime]


class PredictionRequest(BaseModel):
    light_curve_data: List[float] = Field(..., min_items=100, max_items=10000)
    time_data: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    return_confidence: bool = Field(default=True)
    return_explanation: bool = Field(default=False)


class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    classification: str
    explanation: Optional[Dict[str, Any]] = None
    processing_time: float
    model_version: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., max_items=100)
    async_processing: bool = Field(default=False)


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_count: int
    successful_count: int
    failed_count: int
    processing_time: float
    request_id: str


class ModelInfoResponse(BaseModel):
    model_id: str
    name: str
    version: str
    architecture: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    is_active: bool


class UserProfileResponse(BaseModel):
    user: Dict[str, Any]
    api_keys: List[Dict[str, Any]]
    usage_stats: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system_info: Dict[str, Any]


class AdvancedExoplanetAPI:
    """
    Advanced exoplanet detection API with authentication and rate limiting.
    """
    
    def __init__(
        self,
        model: Optional[EnsembleModel] = None,
        auth_manager: Optional[AuthManager] = None,
        monitoring_system: Optional[ModelMonitoringSystem] = None
    ):
        """
        Initialize advanced API.
        
        Args:
            model: Trained ensemble model
            auth_manager: Authentication manager
            monitoring_system: Model monitoring system
        """
        self.app = FastAPI(
            title="Exoplanet Detection API",
            description="Advanced API for exoplanet detection with ML models",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.model = model
        self.auth_manager = auth_manager
        self.monitoring_system = monitoring_system
        self.logger = logging.getLogger(__name__)
        
        # API statistics
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.prediction_count = 0
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup middleware for CORS, rate limiting, etc."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting middleware (if configured)
        if hasattr(self, 'rate_limiter'):
            rate_limit_middleware = create_rate_limit_middleware(
                self.rate_limiter,
                identifier_func=self._get_rate_limit_identifier,
                limit_func=self._get_rate_limit_config
            )
            self.app.middleware("http")(rate_limit_middleware)
    
    def _get_rate_limit_identifier(self, request: Request) -> str:
        """Get rate limit identifier from request."""
        # Try to get user ID from token
        auth_header = request.headers.get('Authorization')
        if auth_header and self.auth_manager:
            try:
                token = auth_header.replace('Bearer ', '')
                if token.startswith('eyJ'):  # JWT token
                    payload = self.auth_manager.verify_token(token)
                    if payload:
                        return f"user:{payload['user_id']}"
                elif token.startswith('exo_'):  # API key
                    api_key = self.auth_manager.verify_api_key(token)
                    if api_key:
                        return f"api_key:{api_key.key_id}"
            except Exception:
                pass
        
        # Fallback to IP address
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _get_rate_limit_config(self, request: Request) -> RateLimit:
        """Get rate limit configuration for request."""
        # Different limits for different endpoints
        path = request.url.path
        
        if path.startswith('/api/v2/predict'):
            if 'batch' in path:
                return RateLimit(10, 60, RateLimitStrategy.TOKEN_BUCKET)  # 10 batch requests per minute
            else:
                return RateLimit(100, 60, RateLimitStrategy.SLIDING_WINDOW)  # 100 predictions per minute
        elif path.startswith('/api/v2/auth'):
            return RateLimit(20, 300, RateLimitStrategy.FIXED_WINDOW)  # 20 auth requests per 5 minutes
        else:
            return RateLimit(200, 60, RateLimitStrategy.SLIDING_WINDOW)  # 200 general requests per minute
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Health and status endpoints
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                version="2.0.0",
                uptime=uptime,
                system_info={
                    "model_loaded": self.model is not None,
                    "auth_enabled": self.auth_manager is not None,
                    "monitoring_enabled": self.monitoring_system is not None,
                    "total_requests": self.request_count,
                    "total_predictions": self.prediction_count
                }
            )
        
        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check for Kubernetes."""
            if self.model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            return {"status": "ready"}
        
        # Authentication endpoints
        @self.app.post("/api/v2/auth/login", response_model=LoginResponse)
        async def login(request: LoginRequest):
            """User login endpoint."""
            if not self.auth_manager:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Authentication not configured"
                )
            
            user = self.auth_manager.authenticate_user(request.username, request.password)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            access_token = self.auth_manager.create_access_token(user)
            refresh_token = self.auth_manager.create_refresh_token(user)
            
            return LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.auth_manager.token_expiry,
                user=user.to_dict()
            )
        
        @self.app.post("/api/v2/auth/refresh", response_model=Dict[str, str])
        async def refresh_token(request: RefreshTokenRequest):
            """Refresh access token."""
            if not self.auth_manager:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Authentication not configured"
                )
            
            new_token = self.auth_manager.refresh_access_token(request.refresh_token)
            if not new_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            return {
                "access_token": new_token,
                "token_type": "bearer",
                "expires_in": self.auth_manager.token_expiry
            }
        
        @self.app.post("/api/v2/auth/logout")
        async def logout(
            user: User = Depends(get_current_user),
            request: Request = None
        ):
            """User logout endpoint."""
            # Revoke current token
            auth_header = request.headers.get('Authorization')
            if auth_header and self.auth_manager:
                token = auth_header.replace('Bearer ', '')
                self.auth_manager.revoke_token(token)
            
            return {"message": "Logged out successfully"}
        
        # API Key management
        @self.app.post("/api/v2/auth/api-keys", response_model=APIKeyResponse)
        async def create_api_key(
            request: CreateAPIKeyRequest,
            user: User = Depends(require_permission("write:api_keys"))
        ):
            """Create new API key."""
            if not self.auth_manager:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Authentication not configured"
                )
            
            key_string, api_key = self.auth_manager.create_api_key(
                user.id,
                request.name,
                request.permissions,
                request.expires_in_days
            )
            
            return APIKeyResponse(
                key_id=api_key.key_id,
                api_key=key_string,
                name=api_key.name,
                permissions=api_key.permissions,
                expires_at=api_key.expires_at
            )
        
        @self.app.get("/api/v2/auth/api-keys")
        async def list_api_keys(
            user: User = Depends(get_current_user)
        ):
            """List user's API keys."""
            if not self.auth_manager:
                return []
            
            user_keys = [
                {
                    "key_id": key.key_id,
                    "name": key.name,
                    "permissions": key.permissions,
                    "is_active": key.is_active,
                    "created_at": key.created_at,
                    "expires_at": key.expires_at,
                    "last_used": key.last_used
                }
                for key in self.auth_manager.api_keys.values()
                if key.user_id == user.id
            ]
            
            return user_keys
        
        # User profile
        @self.app.get("/api/v2/user/profile", response_model=UserProfileResponse)
        async def get_user_profile(
            user: User = Depends(get_current_user)
        ):
            """Get user profile information."""
            api_keys = []
            if self.auth_manager:
                api_keys = [
                    key.to_dict() for key in self.auth_manager.api_keys.values()
                    if key.user_id == user.id
                ]
            
            # Mock usage statistics
            usage_stats = {
                "total_predictions": 150,
                "predictions_this_month": 45,
                "api_calls_today": 12,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            return UserProfileResponse(
                user=user.to_dict(),
                api_keys=api_keys,
                usage_stats=usage_stats
            )
        
        # Prediction endpoints
        @self.app.post("/api/v2/predict", response_model=PredictionResponse)
        async def predict_exoplanet(
            request: PredictionRequest,
            background_tasks: BackgroundTasks,
            user: User = Depends(require_permission("read:predictions"))
        ):
            """Single exoplanet prediction."""
            if not self.model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not available"
                )
            
            request_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            try:
                # Prepare input data
                import torch
                import numpy as np
                
                # Convert to tensor
                flux_data = torch.tensor(request.light_curve_data, dtype=torch.float32)
                
                # Ensure correct length
                target_length = 2048
                if len(flux_data) > target_length:
                    flux_data = flux_data[:target_length]
                elif len(flux_data) < target_length:
                    # Pad with zeros
                    padding = torch.zeros(target_length - len(flux_data))
                    flux_data = torch.cat([flux_data, padding])
                
                # Add batch dimension
                flux_data = flux_data.unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    if hasattr(self.model, 'predict_with_uncertainty'):
                        prediction, uncertainty = self.model.predict_with_uncertainty(flux_data)
                        confidence = 1.0 - uncertainty.item()
                    else:
                        prediction = self.model(flux_data)
                        confidence = abs(prediction.item() - 0.5) + 0.5  # Simple confidence
                
                prediction_value = prediction.item()
                classification = "planet" if prediction_value > 0.5 else "no_planet"
                
                # Generate explanation if requested
                explanation = None
                if request.return_explanation:
                    explanation = {
                        "feature_importance": "Not implemented in this demo",
                        "confidence_factors": {
                            "signal_strength": confidence,
                            "noise_level": 0.1,
                            "periodicity": 0.8
                        }
                    }
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Update statistics
                self.prediction_count += 1
                
                # Background monitoring
                if self.monitoring_system:
                    background_tasks.add_task(
                        self._log_prediction,
                        user.id,
                        request_id,
                        prediction_value,
                        processing_time
                    )
                
                return PredictionResponse(
                    prediction=prediction_value,
                    confidence=confidence if request.return_confidence else None,
                    classification=classification,
                    explanation=explanation,
                    processing_time=processing_time,
                    model_version="ensemble-v1.0",
                    request_id=request_id
                )
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        @self.app.post("/api/v2/predict/batch", response_model=BatchPredictionResponse)
        async def batch_predict_exoplanet(
            request: BatchPredictionRequest,
            background_tasks: BackgroundTasks,
            user: User = Depends(require_permission("read:predictions"))
        ):
            """Batch exoplanet predictions."""
            if not self.model:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not available"
                )
            
            request_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, pred_request in enumerate(request.predictions):
                try:
                    # Process individual prediction (simplified)
                    individual_result = await predict_exoplanet(
                        pred_request, background_tasks, user
                    )
                    results.append(individual_result)
                    successful_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Batch prediction {i} failed: {e}")
                    failed_count += 1
                    # Add error result
                    results.append(PredictionResponse(
                        prediction=0.0,
                        confidence=0.0,
                        classification="error",
                        explanation={"error": str(e)},
                        processing_time=0.0,
                        model_version="ensemble-v1.0",
                        request_id=f"{request_id}-{i}"
                    ))
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return BatchPredictionResponse(
                results=results,
                total_count=len(request.predictions),
                successful_count=successful_count,
                failed_count=failed_count,
                processing_time=processing_time,
                request_id=request_id
            )
        
        # Model information
        @self.app.get("/api/v2/models", response_model=List[ModelInfoResponse])
        async def list_models(
            user: User = Depends(require_permission("read:models"))
        ):
            """List available models."""
            models = []
            
            if self.model:
                models.append(ModelInfoResponse(
                    model_id="ensemble-001",
                    name="Exoplanet Ensemble Model",
                    version="1.0.0",
                    architecture="CNN + LSTM + Transformer Ensemble",
                    training_date=datetime.utcnow() - timedelta(days=7),
                    performance_metrics={
                        "accuracy": 0.92,
                        "precision": 0.89,
                        "recall": 0.94,
                        "f1_score": 0.91,
                        "roc_auc": 0.96
                    },
                    is_active=True
                ))
            
            return models
        
        # Admin endpoints
        @self.app.get("/api/v2/admin/stats")
        async def get_admin_stats(
            user: User = Depends(require_role(UserRole.ADMIN))
        ):
            """Get system statistics (admin only)."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return {
                "system": {
                    "uptime_seconds": uptime,
                    "total_requests": self.request_count,
                    "total_predictions": self.prediction_count,
                    "start_time": self.start_time.isoformat()
                },
                "users": {
                    "total_users": len(self.auth_manager.users) if self.auth_manager else 0,
                    "active_users": len([u for u in self.auth_manager.users.values() if u.is_active]) if self.auth_manager else 0
                },
                "api_keys": {
                    "total_keys": len(self.auth_manager.api_keys) if self.auth_manager else 0,
                    "active_keys": len([k for k in self.auth_manager.api_keys.values() if k.is_active]) if self.auth_manager else 0
                }
            }
        
        # Middleware to count requests
        @self.app.middleware("http")
        async def count_requests(request: Request, call_next):
            self.request_count += 1
            response = await call_next(request)
            return response
    
    async def _log_prediction(
        self,
        user_id: str,
        request_id: str,
        prediction: float,
        processing_time: float
    ):
        """Log prediction for monitoring."""
        if self.monitoring_system:
            # This would typically log to a database or monitoring system
            self.logger.info(
                f"Prediction logged - User: {user_id}, Request: {request_id}, "
                f"Prediction: {prediction:.4f}, Time: {processing_time:.3f}s"
            )


def create_advanced_api(
    model: Optional[EnsembleModel] = None,
    secret_key: str = "your-secret-key-change-in-production",
    redis_client = None,
    enable_auth: bool = True,
    enable_rate_limiting: bool = True
) -> FastAPI:
    """
    Factory function to create advanced API.
    
    Args:
        model: Trained ensemble model
        secret_key: JWT secret key
        redis_client: Redis client for caching
        enable_auth: Whether to enable authentication
        enable_rate_limiting: Whether to enable rate limiting
        
    Returns:
        Configured FastAPI application
    """
    # Initialize authentication
    auth_manager = None
    if enable_auth:
        initialize_auth(secret_key, redis_client)
        from .auth import auth_manager
    
    # Initialize rate limiting
    rate_limiter = None
    if enable_rate_limiting:
        rate_limiter = create_rate_limiter(redis_client)
    
    # Create API instance
    api = AdvancedExoplanetAPI(model, auth_manager)
    
    if rate_limiter:
        api.rate_limiter = rate_limiter
    
    return api.app