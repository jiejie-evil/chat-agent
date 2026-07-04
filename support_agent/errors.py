from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ServiceError(Exception):
    error_code: str
    status_code: int
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_response(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details or {},
            }
        }


class AuthenticationRequiredError(ServiceError):
    def __init__(self):
        super().__init__(
            error_code="authentication_required",
            status_code=401,
            message="X-API-Key header is required.",
        )


class AuthenticationInvalidError(ServiceError):
    def __init__(self):
        super().__init__(
            error_code="authentication_invalid",
            status_code=403,
            message="The provided API key is invalid.",
        )


class RateLimitedError(ServiceError):
    def __init__(self, retry_after_seconds: int):
        super().__init__(
            error_code="rate_limited",
            status_code=429,
            message="Rate limit exceeded for /answer.",
            details={"retry_after_seconds": retry_after_seconds},
        )


class RequestTimeoutError(ServiceError):
    def __init__(self, timeout_seconds: float):
        super().__init__(
            error_code="request_timeout",
            status_code=504,
            message="The request exceeded the allowed processing time.",
            details={"timeout_seconds": timeout_seconds},
        )


class UpstreamUnavailableError(ServiceError):
    def __init__(self):
        super().__init__(
            error_code="upstream_unavailable",
            status_code=503,
            message="The model provider is temporarily unavailable.",
        )
