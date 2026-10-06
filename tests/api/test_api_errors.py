from types import SimpleNamespace

import pytest

from scripts.api import run_experiment


class HttpError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__("HTTP {}".format(status_code))
        self.status_code = status_code
        headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def gerar(self, prompt):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_429_is_retried_using_retry_after(monkeypatch, fake_result):
    sleeps = []
    monkeypatch.setattr(run_experiment.time, "sleep", sleeps.append)
    client = SequenceClient([HttpError(429, retry_after=7), fake_result])

    result, attempts = run_experiment.call_with_retries(client, "prompt", 3, 2.0)

    assert result is fake_result
    assert attempts == 2
    assert client.calls == 2
    assert sleeps == [7.0]


def test_500_uses_exponential_backoff(monkeypatch, fake_result):
    sleeps = []
    monkeypatch.setattr(run_experiment.time, "sleep", sleeps.append)
    client = SequenceClient([HttpError(500), HttpError(503), fake_result])

    _, attempts = run_experiment.call_with_retries(client, "prompt", 3, 2.0)

    assert attempts == 3
    assert sleeps == [2.0, 4.0]


def test_401_is_not_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_experiment.time, "sleep", sleeps.append)
    client = SequenceClient([HttpError(401)])

    with pytest.raises(HttpError):
        run_experiment.call_with_retries(client, "prompt", 3, 2.0)

    assert client.calls == 1
    assert sleeps == []


def test_retry_limit_is_respected(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_experiment.time, "sleep", sleeps.append)
    client = SequenceClient([HttpError(503), HttpError(503), HttpError(503)])

    with pytest.raises(HttpError) as captured:
        run_experiment.call_with_retries(client, "prompt", 2, 1.0)

    assert client.calls == 3
    assert sleeps == [1.0, 2.0]
    assert captured.value.api_attempts == 3
