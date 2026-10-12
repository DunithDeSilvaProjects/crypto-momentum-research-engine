import pandas as pd

import momentum.data as data


def test_fetch_history_pages_through_long_history(monkeypatch):
    full = pd.DataFrame({"close": range(3200)},
                        index=pd.date_range("2021-01-01", periods=3200, freq="D", tz="UTC"))
    calls = []

    def fake_fetch(symbol, interval="1d", limit=1000, start_ms=None):
        calls.append(start_ms)
        start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        return full[full.index >= start].head(limit)

    monkeypatch.setattr(data, "fetch_klines", fake_fetch)

    out = data.fetch_history("BTCUSDT", "2021-01-01")

    assert len(out) == 3200                            # nothing lost
    assert out.index.is_unique                         # nothing duplicated
    assert out.index.is_monotonic_increasing           # in date order
    assert out.index[0] == pd.Timestamp("2021-01-01", tz="UTC")
    assert (out["close"].values == list(range(3200))).all()   # right values on the right days
    assert len(calls) == 4                           # your prediction