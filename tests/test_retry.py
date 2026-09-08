"""Тесты retry/backoff-логики. Задержки не выполняются реально (sleep_fn=no-op),
поэтому тесты быстрые, но проверяют весь путь: число попыток, итоговый успех,
и то, что не-retryable ошибки не ретраятся вовсе."""
import anthropic
import httpx
import pytest

from pdf_qa_agent.retry import RetryConfig, RetryExhaustedError, compute_delay, with_retry


def _fake_connection_error() -> anthropic.APIConnectionError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


def test_succeeds_without_retry_when_no_error():
    calls = []

    @with_retry(RetryConfig(max_retries=3), sleep_fn=lambda d: calls.append(d))
    def func():
        return "ok"

    assert func() == "ok"
    assert calls == []  # ни одной задержки — успех с первой попытки


def test_retries_then_succeeds():
    attempts = {"n": 0}

    @with_retry(RetryConfig(max_retries=5), sleep_fn=lambda d: None)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _fake_connection_error()
        return "recovered"

    assert flaky() == "recovered"
    assert attempts["n"] == 3


def test_raises_retry_exhausted_after_max_attempts():
    attempts = {"n": 0}

    @with_retry(RetryConfig(max_retries=3), sleep_fn=lambda d: None)
    def always_fails():
        attempts["n"] += 1
        raise _fake_connection_error()

    with pytest.raises(RetryExhaustedError) as exc_info:
        always_fails()

    assert attempts["n"] == 3
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, anthropic.APIConnectionError)


def test_non_retryable_exception_propagates_immediately():
    attempts = {"n": 0}

    @with_retry(RetryConfig(max_retries=5), sleep_fn=lambda d: None)
    def raises_value_error():
        attempts["n"] += 1
        raise ValueError("bad input, retrying won't help")

    with pytest.raises(ValueError):
        raises_value_error()

    assert attempts["n"] == 1  # ни одной повторной попытки


def test_on_retry_callback_invoked_with_attempt_and_delay():
    calls = []

    @with_retry(
        RetryConfig(max_retries=2),
        sleep_fn=lambda d: None,
        on_retry=lambda attempt, exc, delay: calls.append((attempt, delay)),
    )
    def always_fails():
        raise _fake_connection_error()

    with pytest.raises(RetryExhaustedError):
        always_fails()

    assert len(calls) == 1  # 2 попытки -> 1 промежуточный retry
    assert calls[0][0] == 1


def test_compute_delay_grows_exponentially_and_respects_cap():
    config = RetryConfig(max_retries=10, base_delay=1.0, max_delay=5.0, jitter=0.0)
    assert compute_delay(1, config) == pytest.approx(1.0)
    assert compute_delay(2, config) == pytest.approx(2.0)
    assert compute_delay(3, config) == pytest.approx(4.0)
    assert compute_delay(10, config) == pytest.approx(5.0)  # capped at max_delay
