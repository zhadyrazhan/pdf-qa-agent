"""Retry / exponential backoff for OpenAI API calls.

Wraps flaky network calls (rate limits, timeouts, transient 5xx errors) with
retries using exponential backoff and jitter, so the agent doesn't fail on a
single network hiccup or momentary API overload.
"""
from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Type, TypeVar

import openai

T = TypeVar("T")

# Errors worth retrying: transient network/server issues and rate limits.
# Request-validation errors (4xx other than 429) aren't retried — a retry
# would just fail the same way again.
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25  # random extra fraction of delay added (0..jitter*delay)


class RetryExhaustedError(RuntimeError):
    """All attempts were exhausted — the original error is kept in last_error."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"Retry exhausted after {attempts} attempt(s): {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


def compute_delay(attempt: int, config: RetryConfig) -> float:
    """Exponential delay with jitter for a given (1-indexed) attempt number."""
    delay = min(config.base_delay * (2 ** (attempt - 1)), config.max_delay)
    return delay + random.uniform(0, config.jitter * delay)


def with_retry(
    config: RetryConfig | None = None,
    retryable_exceptions: Tuple[Type[BaseException], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
):
    """Decorator that retries a call with exponential backoff.

    sleep_fn and on_retry let tests exercise the logic without real delays
    or network access.
    """
    cfg = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: BaseException | None = None
            for attempt in range(1, cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:  # noqa: PERF203 - explicit retry control
                    last_error = exc
                    if attempt == cfg.max_retries:
                        break
                    delay = compute_delay(attempt, cfg)
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    sleep_fn(delay)
            assert last_error is not None
            raise RetryExhaustedError(cfg.max_retries, last_error) from last_error

        return wrapper

    return decorator
