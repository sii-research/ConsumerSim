from __future__ import annotations

from consumersim_mcp_proxy.settings import BackendSettings


def test_url_for_normalizes_paths() -> None:
    settings = BackendSettings(base_url="https://api.example.test")

    assert settings.url_for("forecast") == "https://api.example.test/forecast"
    assert settings.url_for("/forecast") == "https://api.example.test/forecast"


def test_auth_headers_use_bearer_by_default() -> None:
    settings = BackendSettings(base_url="https://api.example.test", api_key="secret")

    assert settings.auth_headers() == {"Authorization": "Bearer secret"}


def test_auth_headers_support_custom_header() -> None:
    settings = BackendSettings(
        base_url="https://api.example.test",
        api_key="secret",
        api_key_header="X-API-Key",
        api_key_scheme="",
    )

    assert settings.auth_headers() == {"X-API-Key": "secret"}
