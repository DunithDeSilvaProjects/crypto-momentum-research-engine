import numpy as np
import pandas as pd
import pytest

from momentum import macro


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_fred_series(monkeypatch):
    payload = {"observations": [
        {"date": "2026-10-12", "value": "4.05"},
        {"date": "2026-10-13", "value": "."},
        {"date": "2026-10-14", "value": "4.10"},
    ]}
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["params"] = params
        return FakeResponse(payload)

    monkeypatch.setattr(macro.requests, "get", fake_get)

    s = macro.fetch_fred_series("DGS10", "KEY", "2026-10-01")

    assert s.name == "DGS10"
    assert s.index[0] == pd.Timestamp("2026-10-12")
    assert s.iloc[0] == pytest.approx(4.05)
    assert np.isnan(s.iloc[1])                # FRED writes "." for a missing day
    assert seen["params"]["series_id"] == "DGS10"


def test_fred_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    assert macro.fred_api_key() == "abc123"


def test_fred_api_key_missing_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FRED_API_KEY environment variable is not set"):
        macro.fred_api_key()

def test_fred_api_key_empty_is_also_an_error(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "")
    with pytest.raises(ValueError):
        macro.fred_api_key()