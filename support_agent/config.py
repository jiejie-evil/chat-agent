import os
from dataclasses import dataclass

from dotenv import load_dotenv


def load_environment() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    load_dotenv(os.path.join(project_root, ".env"))


def _parse_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_api_keys(raw_value: str) -> tuple[str, ...]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(items) if items else ("dev-support-key",)


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    app_env: str
    app_host: str
    app_port: int
    openai_model: str
    database_path: str
    support_agent_api_keys: tuple[str, ...]
    support_agent_require_api_key: bool
    support_agent_request_timeout_seconds: float
    support_agent_openai_timeout_seconds: float
    support_agent_openai_max_retries: int
    support_agent_rate_limit_requests: int
    support_agent_rate_limit_window_seconds: int
    embedding_model: str
    chroma_persist_dir: str


def get_settings() -> Settings:
    load_environment()
    default_db_path = os.path.join(os.path.dirname(__file__), "runtime", "support_agent.db")
    default_chroma_dir = os.path.join(os.path.dirname(__file__), "runtime", "chroma")
    return Settings(
        app_name=os.getenv("APP_NAME", "Support Agent API"),
        app_version=os.getenv("APP_VERSION", "0.3.0"),
        app_env=os.getenv("APP_ENV", "development"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        database_path=os.getenv("SUPPORT_AGENT_DB_PATH", default_db_path),
        support_agent_api_keys=_parse_api_keys(os.getenv("SUPPORT_AGENT_API_KEYS", "")),
        support_agent_require_api_key=_parse_bool(
            os.getenv("SUPPORT_AGENT_REQUIRE_API_KEY", "true"),
            default=True,
        ),
        support_agent_request_timeout_seconds=float(
            os.getenv("SUPPORT_AGENT_REQUEST_TIMEOUT_SECONDS", "20")
        ),
        support_agent_openai_timeout_seconds=float(
            os.getenv("SUPPORT_AGENT_OPENAI_TIMEOUT_SECONDS", "10")
        ),
        support_agent_openai_max_retries=int(
            os.getenv("SUPPORT_AGENT_OPENAI_MAX_RETRIES", "2")
        ),
        support_agent_rate_limit_requests=int(
            os.getenv("SUPPORT_AGENT_RATE_LIMIT_REQUESTS", "10")
        ),
        support_agent_rate_limit_window_seconds=int(
            os.getenv("SUPPORT_AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", default_chroma_dir),
    )
