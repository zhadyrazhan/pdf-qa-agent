"""Retry / exponential backoff для вызовов Anthropic API.

Оборачивает нестабильные сетевые вызовы (rate limit, таймауты, временные 5xx ошибки
сервера) в повторные попытки с экспоненциально растущей задержкой и джиттером, чтобы
агент не падал из-за единичного сбоя сети или временной перегрузки API.
"""
from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Type, TypeVar

import anthropic

T = TypeVar("T")

# Ошибки, при которых имеет смысл повторить запрос: временные сетевые/серверные
# проблемы и rate limit. Ошибки валидации запроса (4xx кроме 429) не ретраятся —
# повторный запрос всё равно завершится той же ошибкой.
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25  # доля от delay, добавляемая случайным образом (0..jitter*delay)


class RetryExhaustedError(RuntimeError):
    """Все попытки исчерпаны — исходная ошибка сохранена в last_error."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"Retry exhausted after {attempts} attempt(s): {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


def compute_delay(attempt: int, config: RetryConfig) -> float:
    """Экспоненциальная задержка с джиттером для номера попытки (1-indexed)."""
    delay = min(config.base_delay * (2 ** (attempt - 1)), config.max_delay)
    return delay + random.uniform(0, config.jitter * delay)


def with_retry(
    config: RetryConfig | None = None,
    retryable_exceptions: Tuple[Type[BaseException], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
):
    """Декоратор с повторными попытками и экспоненциальным backoff.

    Параметры sleep_fn и on_retry позволяют тестировать логику без реальных
    задержек и без обращения к сети.
    """
    cfg = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: BaseException | None = None
            for attempt in range(1, cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:  # noqa: PERF203 - явный контроль ретраев
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
