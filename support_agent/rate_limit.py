import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    bucket_type: str


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, bucket_type: str) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            self._evict_expired(bucket, now)
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    bucket_type=bucket_type,
                )
            bucket.append(now)
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                bucket_type=bucket_type,
            )

    def _evict_expired(self, bucket: deque[float], now: float) -> None:
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()
