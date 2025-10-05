"""
Rate limiting system for the exoplanet detection API.
Provides flexible rate limiting with different strategies and storage backends.
"""

import time
import redis
import json
from typing import Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import hashlib


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int  # Number of requests
    window: int    # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    
    def __str__(self) -> str:
        return f"{self.requests}/{self.window}s"


@dataclass
class RateLimitResult:
    """Rate limit check result."""
    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


class RateLimitStorage(ABC):
    """Abstract base class for rate limit storage."""
    
    @abstractmethod
    async def get_count(self, key: str, window: int) -> int:
        """Get current request count for key."""
        pass
    
    @abstractmethod
    async def increment(self, key: str, window: int) -> int:
        """Increment request count and return new count."""
        pass
    
    @abstractmethod
    async def get_bucket_state(self, key: str) -> Dict[str, Any]:
        """Get token bucket state."""
        pass
    
    @abstractmethod
    async def update_bucket_state(self, key: str, state: Dict[str, Any], ttl: int):
        """Update token bucket state."""
        pass


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage."""
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize Redis storage.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def get_count(self, key: str, window: int) -> int:
        """Get current request count for key."""
        try:
            count = self.redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            self.logger.error(f"Redis get_count error: {e}")
            return 0
    
    async def increment(self, key: str, window: int) -> int:
        """Increment request count and return new count."""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            return results[0]
        except Exception as e:
            self.logger.error(f"Redis increment error: {e}")
            return 1
    
    async def get_bucket_state(self, key: str) -> Dict[str, Any]:
        """Get token bucket state."""
        try:
            state_json = self.redis.get(f"bucket:{key}")
            if state_json:
                return json.loads(state_json)
            return {}
        except Exception as e:
            self.logger.error(f"Redis get_bucket_state error: {e}")
            return {}
    
    async def update_bucket_state(self, key: str, state: Dict[str, Any], ttl: int):
        """Update token bucket state."""
        try:
            self.redis.setex(f"bucket:{key}", ttl, json.dumps(state))
        except Exception as e:
            self.logger.error(f"Redis update_bucket_state error: {e}")


class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage (for testing/development)."""
    
    def __init__(self):
        """Initialize memory storage."""
        self.counts: Dict[str, Tuple[int, float]] = {}  # key -> (count, timestamp)
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def get_count(self, key: str, window: int) -> int:
        """Get current request count for key."""
        if key not in self.counts:
            return 0
        
        count, timestamp = self.counts[key]
        if time.time() - timestamp > window:
            del self.counts[key]
            return 0
        
        return count
    
    async def increment(self, key: str, window: int) -> int:
        """Increment request count and return new count."""
        current_time = time.time()
        
        if key not in self.counts:
            self.counts[key] = (1, current_time)
            return 1
        
        count, timestamp = self.counts[key]
        if current_time - timestamp > window:
            self.counts[key] = (1, current_time)
            return 1
        
        new_count = count + 1
        self.counts[key] = (new_count, timestamp)
        return new_count
    
    async def get_bucket_state(self, key: str) -> Dict[str, Any]:
        """Get token bucket state."""
        return self.buckets.get(key, {})
    
    async def update_bucket_state(self, key: str, state: Dict[str, Any], ttl: int):
        """Update token bucket state."""
        self.buckets[key] = state


class RateLimiter:
    """
    Rate limiter with multiple strategies.
    """
    
    def __init__(
        self,
        storage: RateLimitStorage,
        default_limit: RateLimit = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            storage: Storage backend for rate limit data
            default_limit: Default rate limit configuration
        """
        self.storage = storage
        self.default_limit = default_limit or RateLimit(100, 60)  # 100 requests per minute
        self.logger = logging.getLogger(__name__)
    
    def _get_key(self, identifier: str, limit: RateLimit) -> str:
        """Generate storage key for rate limit."""
        key_data = f"{identifier}:{limit.requests}:{limit.window}:{limit.strategy.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: Optional[RateLimit] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (IP, user ID, API key, etc.)
            limit: Rate limit configuration
            
        Returns:
            Rate limit check result
        """
        if limit is None:
            limit = self.default_limit
        
        if limit.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(identifier, limit)
        elif limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(identifier, limit)
        elif limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(identifier, limit)
        elif limit.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._check_leaky_bucket(identifier, limit)
        else:
            raise ValueError(f"Unknown rate limit strategy: {limit.strategy}")
    
    async def _check_fixed_window(
        self,
        identifier: str,
        limit: RateLimit
    ) -> RateLimitResult:
        """Check fixed window rate limit."""
        key = self._get_key(identifier, limit)
        current_time = int(time.time())
        window_start = (current_time // limit.window) * limit.window
        window_key = f"{key}:{window_start}"
        
        count = await self.storage.increment(window_key, limit.window)
        
        allowed = count <= limit.requests
        remaining = max(0, limit.requests - count)
        reset_time = window_start + limit.window
        retry_after = reset_time - current_time if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def _check_sliding_window(
        self,
        identifier: str,
        limit: RateLimit
    ) -> RateLimitResult:
        """Check sliding window rate limit."""
        # Simplified sliding window using multiple fixed windows
        current_time = int(time.time())
        window_size = limit.window // 4  # Use 4 sub-windows
        
        total_count = 0
        for i in range(4):
            window_start = ((current_time - i * window_size) // window_size) * window_size
            window_key = f"{self._get_key(identifier, limit)}:sliding:{window_start}"
            count = await self.storage.get_count(window_key, window_size)
            total_count += count
        
        # Increment current window
        current_window_start = (current_time // window_size) * window_size
        current_window_key = f"{self._get_key(identifier, limit)}:sliding:{current_window_start}"
        await self.storage.increment(current_window_key, window_size)
        total_count += 1
        
        allowed = total_count <= limit.requests
        remaining = max(0, limit.requests - total_count)
        reset_time = current_time + limit.window
        retry_after = window_size if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def _check_token_bucket(
        self,
        identifier: str,
        limit: RateLimit
    ) -> RateLimitResult:
        """Check token bucket rate limit."""
        key = self._get_key(identifier, limit)
        current_time = time.time()
        
        # Get current bucket state
        state = await self.storage.get_bucket_state(key)
        
        # Initialize bucket if not exists
        if not state:
            state = {
                'tokens': limit.requests,
                'last_refill': current_time
            }
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - state['last_refill']
        tokens_to_add = int(time_elapsed * (limit.requests / limit.window))
        
        # Update tokens (cap at bucket size)
        state['tokens'] = min(limit.requests, state['tokens'] + tokens_to_add)
        state['last_refill'] = current_time
        
        # Check if request can be served
        allowed = state['tokens'] >= 1
        
        if allowed:
            state['tokens'] -= 1
        
        # Update bucket state
        await self.storage.update_bucket_state(key, state, limit.window * 2)
        
        remaining = int(state['tokens'])
        reset_time = int(current_time + (limit.requests - state['tokens']) * (limit.window / limit.requests))
        retry_after = int((1 - state['tokens']) * (limit.window / limit.requests)) if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def _check_leaky_bucket(
        self,
        identifier: str,
        limit: RateLimit
    ) -> RateLimitResult:
        """Check leaky bucket rate limit."""
        key = self._get_key(identifier, limit)
        current_time = time.time()
        
        # Get current bucket state
        state = await self.storage.get_bucket_state(key)
        
        # Initialize bucket if not exists
        if not state:
            state = {
                'level': 0,
                'last_leak': current_time
            }
        
        # Calculate leak based on time elapsed
        time_elapsed = current_time - state['last_leak']
        leak_amount = time_elapsed * (limit.requests / limit.window)
        
        # Update bucket level (minimum 0)
        state['level'] = max(0, state['level'] - leak_amount)
        state['last_leak'] = current_time
        
        # Check if request can be added
        allowed = state['level'] < limit.requests
        
        if allowed:
            state['level'] += 1
        
        # Update bucket state
        await self.storage.update_bucket_state(key, state, limit.window * 2)
        
        remaining = max(0, limit.requests - int(state['level']))
        reset_time = int(current_time + state['level'] * (limit.window / limit.requests))
        retry_after = int((state['level'] - limit.requests + 1) * (limit.window / limit.requests)) if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.
    """
    
    def __init__(
        self,
        rate_limiter: RateLimiter,
        identifier_func: Optional[Callable[[Request], str]] = None,
        limit_func: Optional[Callable[[Request], RateLimit]] = None,
        skip_func: Optional[Callable[[Request], bool]] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            rate_limiter: Rate limiter instance
            identifier_func: Function to extract identifier from request
            limit_func: Function to get rate limit for request
            skip_func: Function to determine if rate limiting should be skipped
        """
        self.rate_limiter = rate_limiter
        self.identifier_func = identifier_func or self._default_identifier
        self.limit_func = limit_func or self._default_limit
        self.skip_func = skip_func or self._default_skip
        self.logger = logging.getLogger(__name__)
    
    def _default_identifier(self, request: Request) -> str:
        """Default identifier extraction (client IP)."""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to client IP
        return request.client.host if request.client else 'unknown'
    
    def _default_limit(self, request: Request) -> RateLimit:
        """Default rate limit configuration."""
        return self.rate_limiter.default_limit
    
    def _default_skip(self, request: Request) -> bool:
        """Default skip logic (never skip)."""
        return False
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Check if rate limiting should be skipped
        if self.skip_func(request):
            return await call_next(request)
        
        # Get identifier and rate limit
        identifier = self.identifier_func(request)
        limit = self.limit_func(request)
        
        # Check rate limit
        try:
            result = await self.rate_limiter.check_rate_limit(identifier, limit)
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            # Continue without rate limiting on error
            return await call_next(request)
        
        # Add rate limit headers to response
        async def add_headers(response):
            response.headers['X-RateLimit-Limit'] = str(limit.requests)
            response.headers['X-RateLimit-Remaining'] = str(result.remaining)
            response.headers['X-RateLimit-Reset'] = str(result.reset_time)
            response.headers['X-RateLimit-Window'] = str(limit.window)
            
            if result.retry_after:
                response.headers['Retry-After'] = str(result.retry_after)
            
            return response
        
        # Check if request is allowed
        if not result.allowed:
            error_response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {limit}',
                    'retry_after': result.retry_after
                }
            )
            return await add_headers(error_response)
        
        # Process request
        response = await call_next(request)
        return await add_headers(response)


def create_rate_limiter(
    redis_client: Optional[redis.Redis] = None,
    default_limit: Optional[RateLimit] = None
) -> RateLimiter:
    """
    Factory function to create rate limiter.
    
    Args:
        redis_client: Redis client for storage
        default_limit: Default rate limit
        
    Returns:
        Configured rate limiter
    """
    if redis_client:
        storage = RedisRateLimitStorage(redis_client)
    else:
        storage = MemoryRateLimitStorage()
    
    return RateLimiter(storage, default_limit)


def create_rate_limit_middleware(
    rate_limiter: RateLimiter,
    **kwargs
) -> RateLimitMiddleware:
    """
    Factory function to create rate limit middleware.
    
    Args:
        rate_limiter: Rate limiter instance
        **kwargs: Additional middleware configuration
        
    Returns:
        Configured middleware
    """
    return RateLimitMiddleware(rate_limiter, **kwargs)