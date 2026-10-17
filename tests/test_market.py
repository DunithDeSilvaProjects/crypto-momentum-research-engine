import pandas as pd
import pytest

from momentum import market


def test_fetch_yf_closes_returns_closes_renamed_to_our_names(monkeypatch):
    idx = pd.to_datetime(["2026-10-15", "2026-10-16"])
    cols = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["DX-Y.NYB", "GC=F"]], names=["Price", "Ticker"])
    fake = pd.DataFrame([[102.0, 4000.0, 0, 5],
                         [101.84, 4010.0, 0, 6]], index=idx, columns=cols)
    monkeypatch.setattr(market.yf, "download", lambda *a, **k: fake)

    out = market.fetch_yf_closes({"DXY": "DX-Y.NYB", "GOLD": "GC=F"}, start="2026-10-01")

    assert list(out.columns) == ["DXY", "GOLD"]
    assert out.shape == (2, 2)
    assert out.loc["2026-10-16", "DXY"] == pytest.approx(101.84)
    assert out["GOLD"].iloc[0] == pytest.approx(4000)


def test_fetch_yf_closes_raises_when_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(market.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError, match="no data"):
        market.fetch_yf_closes({"DXY": "DX-Y.NYB"}, start="2026-10-01")