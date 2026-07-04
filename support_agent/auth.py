import hashlib
from dataclasses import dataclass

from support_agent.config import Settings
from support_agent.errors import AuthenticationInvalidError, AuthenticationRequiredError


@dataclass(frozen=True)
class AuthContext:
    subject: str
    key_fingerprint: str


def fingerprint_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def authenticate_api_key(api_key: str | None, settings: Settings) -> AuthContext:
    if not settings.support_agent_require_api_key:
        return AuthContext(subject="anonymous", key_fingerprint="anonymous")

    if not api_key:
        raise AuthenticationRequiredError()

    if api_key not in settings.support_agent_api_keys:
        raise AuthenticationInvalidError()

    key_fingerprint = fingerprint_api_key(api_key)
    return AuthContext(
        subject=f"apikey:{key_fingerprint}",
        key_fingerprint=key_fingerprint,
    )
