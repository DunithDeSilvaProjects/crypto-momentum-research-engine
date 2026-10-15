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


def test_load_fred_aligns_series_by_date(monkeypatch,tmp_path):

  
    fakes = {
        "DGS10": pd.Series([4.05, 4.07, 4.10], name="DGS10",
                           index=pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14"])),
        "DGS2": pd.Series([3.90, 3.95], name="DGS2",
                          index=pd.to_datetime(["2026-10-12", "2026-10-14"])),
    }
    seen = []

    def fake_get(sid, start, cache_dir):
        seen.append((sid, str(cache_dir)))
        return fakes[sid]

    monkeypatch.setattr(macro, "get_fred_series", fake_get)

    df = macro.load_fred(["DGS10", "DGS2"], "2026-10-01", cache_dir=tmp_path)

    assert list(df.columns) == ["DGS10", "DGS2"]
    assert df.shape == (3, 2)
    assert np.isnan(df.loc["2026-10-13", "DGS2"])
    assert df.loc["2026-10-14", "DGS2"] == pytest.approx(3.95)
    assert seen[0] == ("DGS10", str(tmp_path))      # the first series asked for, and the cache folder passed down

def test_get_fred_series_refetches_a_week_and_keeps_newest(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "KEY")
    calls = []
    first = pd.Series([4.05, 4.07, np.nan], name="DGS10",
                      index=pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14"]))
    second = pd.Series([4.10, 4.12], name="DGS10",
                       index=pd.to_datetime(["2026-10-14", "2026-10-15"]))
    replies = [first, second]

    def fake_fetch(sid, key, start):
        calls.append(start)
        return replies[len(calls) - 1]

    monkeypatch.setattr(macro, "fetch_fred_series", fake_fetch)

    macro.get_fred_series("DGS10", "2020-01-01", cache_dir=tmp_path)
    s = macro.get_fred_series("DGS10", "2020-01-01", cache_dir=tmp_path)

    assert calls[0] == "2020-01-01"                            # first run: no cache yet
    assert calls[1] == "2026-10-07"                            # second run: a week before the last cached date
    assert len(s) == 4                                 # 12, 13, 14 and 15, with no duplicates
    assert s.loc["2026-10-14"] == pytest.approx(4.10)    # the new value replaced the NaN
    assert s.loc["2026-10-12"] == pytest.approx(4.05)    # this one comes only from the cache