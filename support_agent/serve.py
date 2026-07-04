import uvicorn

from support_agent.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "support_agent.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    main()
