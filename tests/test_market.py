import pandas as pd
import pytest
import numpy as np
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

def test_last_complete_session_is_the_latest_closed_us_trading_day():
    def last(utc):
        return market.last_complete_session(pd.Timestamp(utc, tz="UTC"))

    assert last("2026-10-19 16:55") == pd.Timestamp("2026-10-16")    # Mon, US market open
    assert last("2026-10-20 00:30") == pd.Timestamp("2026-10-19")    # the daily workflow run
    assert last("2026-10-19 07:00") == pd.Timestamp("2026-10-16")    # Mon, before the US open
    assert last("2026-10-17 12:00") == pd.Timestamp("2026-10-16")    # Saturday
    assert last("2026-10-16 22:00") == pd.Timestamp("2026-10-16")    # Fri, after the close
    assert last("2026-10-18 23:00") == pd.Timestamp("2026-10-16")    # Sunday evening

def test_drop_incomplete_removes_rows_after_the_cutoff_and_keeps_the_cutoff_day():
    idx = pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"])
    df = pd.DataFrame({"VIX": [19.0, 20.0, 20.5, 18.1],
                       "GOLD": [4000.0, 4010.0, 4020.0, 4030.0]}, index=idx)

    out = market.drop_incomplete(df, pd.Timestamp("2026-10-16"))

    assert list(out.index) == [pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-16")]
    assert out["VIX"].iloc[-1] == 20.5
    assert out.shape == (3, 2)
    assert len(df) == 4                                          # the original is untouched
    assert market.drop_incomplete(df, pd.Timestamp("2026-10-13")).empty

def test_render_asof_shows_generation_time_in_three_zones_and_the_cutoff():
    now = pd.Timestamp("2026-10-20 00:34", tz="UTC")
    lines = market.render_asof(now, pd.Timestamp("2026-10-19")).split("\n")

    assert lines[0] == "Generated: 20 Oct 00:34 UTC · 20 Oct 08:34 Singapore/Hong Kong · 19 Oct 20:34 New York"
    assert lines[1] == "Market data through last complete US session: Mon 19 Oct 2026 (close 16:00 ET)"


def test_render_asof_new_york_time_follows_daylight_saving():
    now = pd.Timestamp("2026-11-02 00:34", tz="UTC")
    text = market.render_asof(now, pd.Timestamp("2026-10-30"))

    assert "01 Nov 19:34 New York" in text

def test_fetch_yf_optional_returns_the_data_and_no_note_when_the_fetch_works():
    good = pd.DataFrame({"VIX": [19.0]}, index=pd.to_datetime(["2026-10-16"]))

    out, note = market.fetch_yf_optional({"VIX": "^VIX"}, "2025-01-01",
                                         fetch=lambda symbols, start: good)

    assert out is good
    assert note is None


def test_fetch_yf_optional_returns_an_empty_table_and_a_note_when_the_fetch_fails():
    def broken(symbols, start):
        raise RuntimeError("boom")

    out, note = market.fetch_yf_optional({"VIX": "^VIX"}, "2025-01-01", fetch=broken)

    assert out.empty
    assert note == "Yahoo Finance data unavailable today (RuntimeError)"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert market.drop_incomplete(out, pd.Timestamp("2026-10-16")).empty    # must not raise


def test_refill_empty_requests_only_empty_or_missing_tickers():
    index = pd.to_datetime(["2026-10-22", "2026-10-23"])
    closes = pd.DataFrame({"SPX": [1.0, 2.0], "NK": [np.nan, np.nan]}, index=index)
    symbols = {"SPX": "^GSPC", "NK": "^N225", "EU": "^STOXX50E"}
    asked = []

    def fake_download(ticker, start):
        asked.append(ticker)
        if ticker == "^N225":
            return pd.Series([10.0, 11.0], index=index)
        return pd.Series(dtype=float)

    out = market.refill_empty(closes, symbols, "2026-01-01", fake_download)

    assert asked == ['^N225', '^STOXX50E']
    assert out["NK"].tolist() == [10.0, 11.0]
    assert out["SPX"].tolist() == [1.0, 2.0]
    assert "EU" not in out.columns

def test_fetch_yf_closes_refills_a_ticker_the_batch_returned_empty(monkeypatch):
    index = pd.to_datetime(["2026-10-22", "2026-10-23"])
    batch = pd.concat(
        {"Close": pd.DataFrame({"^GSPC": [1.0, 2.0], "^N225": [np.nan, np.nan]}, index=index)},
        axis=1,
    )
    monkeypatch.setattr(market.yf, "download", lambda *args, **kwargs: batch)

    out = market.fetch_yf_closes(
        {"SPX": "^GSPC", "NK": "^N225"}, "2026-01-01",
        download=lambda ticker, start: pd.Series([10.0, 11.0], index=index),
    )

    assert list(out.columns) == ["SPX", "NK"]
    assert out["SPX"].tolist() == [1.0, 2.0]
    assert out["NK"].tolist() == [10, 11]


def test_prefer_fresh_takes_fresh_values_and_falls_back_for_older_dates():
    fred = pd.Series([100.0, 101.5], index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    yahoo = pd.Series(
        [101.0, 102.0, 103.0],
        index=pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"]),
    )

    out = market.prefer_fresh(yahoo, fred)

    assert out.tolist() == [100.0, 101.0, 102.0, 103.0]
    assert out.index[-1] == pd.Timestamp("2026-10-23")
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert market.prefer_fresh(empty, fred).tolist() == [100.0, 101.5]


def test_overlay_fresh_extends_a_column_and_ignores_unmapped_ones():
    base = pd.DataFrame(
        {"SP500": [100.0, 101.5], "DGS10": [4.9, 4.95]},
        index=pd.to_datetime(["2026-10-20", "2026-10-21"]),
    )
    fresh = pd.DataFrame(
        {"SP500_YF": [101.0, 102.0, 103.0], "VIX": [20.0, 21.0, 22.0]},
        index=pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"]),
    )

    out = market.overlay_fresh(base, fresh, {"SP500": "SP500_YF"})

    assert out["SP500"].tolist() == [100.0, 101.0, 102.0, 103.0]
    assert out["DGS10"].tolist()[:2] == [4.9, 4.95]
    assert out["DGS10"].isna().tolist() == [False, False, True, True]
    assert "VIX" not in out.columns
    down = pd.DataFrame(index=pd.DatetimeIndex([]))
    assert market.overlay_fresh(base, down, {"SP500": "SP500_YF"})["SP500"].tolist() == [100.0, 101.5]