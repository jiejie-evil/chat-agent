import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from support_agent.errors import RequestTimeoutError, UpstreamUnavailableError


LOGGER = logging.getLogger("support_agent")

T = TypeVar("T")


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def log_event(payload: dict) -> None:
    configure_logging()
    LOGGER.info(json.dumps(payload, ensure_ascii=True))


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float) -> T:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RequestTimeoutError(timeout_seconds) from exc


def run_with_retries(
    fn: Callable[[], T],
    max_retries: int,
    on_retry: Callable[[int, float], None] | None = None,
    on_exhausted: Callable[[], None] | None = None,
) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries:
                if on_exhausted:
                    on_exhausted()
                raise UpstreamUnavailableError() from exc
            delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.1)
            if on_retry:
                on_retry(attempt + 1, delay)
            time.sleep(delay)
            attempt += 1
