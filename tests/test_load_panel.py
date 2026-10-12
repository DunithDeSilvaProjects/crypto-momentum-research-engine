import numpy as np
import pandas as pd

import momentum.data as data


def fake_frame(dates, closes):
    idx = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({"close": closes,
                         "quote_volume": [1000.0] * len(closes),
                         "taker_buy_quote": [500.0] * len(closes)}, index=idx)


def test_load_panel_aligns_by_date_and_leaves_nan_before_listing(monkeypatch):
    fakes = {
        "BTCUSDT": fake_frame(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 105.0]),
        "SOLUSDT": fake_frame(["2024-01-02", "2024-01-03"], [20.0, 22.0]),
    }
    monkeypatch.setattr(data, "get_history", lambda s, start: fakes[s])

    p = data.load_panel(["BTCUSDT", "SOLUSDT"], "2024-01-01")

    assert list(p.close.columns) == ["BTCUSDT", "SOLUSDT"]
    assert len(p.close) == 3
    assert np.isnan(p.close["SOLUSDT"].iloc[0])        # SOL had not listed yet
    assert p.close["SOLUSDT"].iloc[1] == 20.0
    assert p.quote_volume["BTCUSDT"].iloc[0] == 1000.0
    assert p.funding.shape == p.close.shape
    assert p.funding.isna().all().all()                # no funding data yet: must be empty, not buy volume


def test_build_default_panel_uses_the_fixed_universe(monkeypatch):
    seen = {}

    def fake_load_panel(symbols, start):
        seen["symbols"] = symbols
        seen["start"] = start
        return "sentinel"

    monkeypatch.setattr(data, "load_panel", fake_load_panel)

    out = data.build_default_panel("2022-05-01")

    assert out == "sentinel"
    assert seen["symbols"] == data.DEFAULT_SYMBOLS
    assert seen["start"] == "2022-05-01"