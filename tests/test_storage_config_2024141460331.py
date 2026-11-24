"""Storage selection must fail closed without falling back to HTTP."""

import pytest

from a4_speech.cli_2024141460331 import (
    ConfigError_2024141460331,
    _api_context_2024141460331,
)


def test_unknown_storage_mode_is_rejected():
    with pytest.raises(ConfigError_2024141460331):
        with _api_context_2024141460331({"storage": {"mode": "typo"}}):
            pytest.fail("an invalid mode must not create an adapter")


def test_postgres_requires_dsn_environment_and_audio_root(monkeypatch, tmp_path):
    monkeypatch.delenv("A4_DATABASE_URL", raising=False)
    config = {"_directory": str(tmp_path), "storage": {"mode": "postgres", "audioRoot": "."}}
    with pytest.raises(ConfigError_2024141460331):
        with _api_context_2024141460331(config):
            pytest.fail("database mode must require its own credential")


def test_default_http_still_requires_upstream_credential(monkeypatch):
    monkeypatch.delenv("A4_API_KEY", raising=False)
    config = {"a3Url": "http://localhost/a3", "a4Url": "http://localhost/a4"}
    with pytest.raises(ConfigError_2024141460331, match="AUTH_ENV_MISSING"):
        with _api_context_2024141460331(config):
            pytest.fail("HTTP must retain its authentication boundary")


@pytest.mark.parametrize("code", ["POSTGRES_UNAVAILABLE", "REVISION_AUTH_UNAVAILABLE"])
def test_local_storage_unavailable_is_service_unavailable(code):
    from a4_speech.api_2024141460331 import _error_status_2024141460331
    from a4_speech.service_2024141460331 import ProcessingError_2024141460331

    error = ProcessingError_2024141460331(code, http_status=503)
    assert _error_status_2024141460331(error) == 503
