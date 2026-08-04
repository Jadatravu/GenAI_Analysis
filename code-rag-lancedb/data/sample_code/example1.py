"""Utilities for resilient HTTP calls: retries, exponential backoff, and a
simple in-memory rate limiter."""
import random
import time
from dataclasses import dataclass, field


def exponential_backoff(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    """Compute an exponential backoff delay with jitter for a given attempt number."""
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


def retry_with_backoff(func, max_attempts: int = 5, base: float = 0.5):
    """Call `func` with no arguments, retrying on exception with exponential backoff."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(exponential_backoff(attempt, base=base))
    raise RuntimeError(f"All {max_attempts} attempts failed") from last_exc


@dataclass
class RateLimiter:
    """Token-bucket style in-memory rate limiter."""

    max_calls: int
    period_seconds: float
    _calls: list = field(default_factory=list)

    def allow(self) -> bool:
        """Return True if a call is allowed right now, recording it if so."""
        now = time.time()
        self._calls = [t for t in self._calls if now - t < self.period_seconds]
        if len(self._calls) < self.max_calls:
            self._calls.append(now)
            return True
        return False

    def wait_time(self) -> float:
        """Seconds to wait before the next call would be allowed."""
        if not self._calls:
            return 0.0
        oldest = min(self._calls)
        return max(0.0, self.period_seconds - (time.time() - oldest))
