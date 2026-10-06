import pandas as pd
import pytest
import requests

import momentum.data as data


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class ErrorResponse:
    def raise_for_status(self):
        raise requests.HTTPError("boom")


def candle(open_ms, close_ms, close_price):
    # open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
    return [open_ms, "100", "110", "90", str(close_price), "10", close_ms, "1000", 5, "4", "500", "0"]


def test_unfinished_candle_is_dropped(monkeypatch):
    now = pd.Timestamp("2024-01-06 12:00", tz="UTC")
    day = 86_400_000
    day1 = int(pd.Timestamp("2024-01-05", tz="UTC").timestamp() * 1000)
    day2 = day1 + day
    payload = [candle(day1, day2 - 1, 105), candle(day2, day2 + day - 1, 108)]

    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(payload))
    monkeypatch.setattr(data.pd.Timestamp, "now", staticmethod(lambda tz=None: now))

    out = data.fetch_klines("BTCUSDT")
    assert len(out) == 1
    assert out["close"].iloc[0] == 105.0


def test_http_error_raises(monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: ErrorResponse())

    with pytest.raises(requests.HTTPError) as exc_info:
        data.fetch_klines("BTCUSDT")

    assert "boom" in str(exc_info.value)