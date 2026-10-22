import pandas as pd

from momentum import crypto
import pytest


def test_parse_klines_keeps_finished_days_and_drops_the_unfinished_one():
    def row(open_ms, close, quote_volume):
        return [open_ms, "0", "0", "0", close, "0", open_ms + 86_399_999,
                quote_volume, 0, "0", "0", "0"]

    rows = [row(1_699_920_000_000, "105", "1050"),    # 14 Nov 2023
            row(1_700_006_400_000, "110", "2200"),    # 15 Nov 2023
            row(1_700_092_800_000, "112", "560")]     # 16 Nov 2023, still running
    now = pd.Timestamp("2023-11-16 12:00", tz="UTC")

    out = crypto.parse_klines(rows, now)

    assert list(out.index) == [pd.Timestamp("14 Nov 2023"), pd.Timestamp("15 Nov 2023")]
    assert out["close"].tolist() == [105, 110]
    assert out["quote_volume"].iloc[-1] == 2200
    assert out.index.tz is None

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_fetch_spot_klines_asks_the_public_data_host_for_daily_candles():
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse([["row"]])

    rows = crypto.fetch_spot_klines("BTCUSDT", limit=400, get=fake_get)

    assert rows == [["row"]]
    url, params, timeout = calls[0]
    assert url == "https://data-api.binance.vision/api/v3/klines"
    assert params == {"symbol": "BTCUSDT", "interval": "1d", "limit": 400}
    assert timeout == 30


def test_fetch_spot_klines_raises_when_the_server_refuses():
    class Refused(FakeResponse):
        def raise_for_status(self):
            raise RuntimeError("451")

    with pytest.raises(RuntimeError):
        crypto.fetch_spot_klines("BTCUSDT", get=lambda url, params, timeout: Refused([]))

NOW = pd.Timestamp("2023-11-16 12:00", tz="UTC")


def kline_row(open_ms, close, quote_volume):
    return [open_ms, "0", "0", "0", close, "0", open_ms + 86_399_999,
            quote_volume, 0, "0", "0", "0"]


def test_load_spot_panel_builds_aligned_tables_and_reports_the_coins_that_failed():
    data = {
        "AAA": [kline_row(1_699_920_000_000, "10", "100"),
                kline_row(1_700_006_400_000, "11", "110")],
        "CCC": [kline_row(1_699_920_000_000, "20", "200"),
                kline_row(1_700_006_400_000, "22", "220")],
    }

    def fake_fetch(symbol):
        if symbol == "BBB":
            raise RuntimeError("451")
        return data[symbol]

    closes, volumes, failed = crypto.load_spot_panel(["AAA", "BBB", "CCC"], NOW, fetch=fake_fetch)

    assert list(closes.columns) == ["AAA", "CCC"]
    assert failed == ["BBB"]
    assert closes.loc["2023-11-15", "CCC"] == 22
    assert volumes["AAA"].tolist() == [100, 110]
    assert closes.shape == (2, 2)

def test_load_spot_panel_with_every_coin_failing_returns_empty_date_indexed_tables():
    def broken(symbol):
        raise RuntimeError("451")

    closes, volumes, failed = crypto.load_spot_panel(["AAA", "BBB"], NOW, fetch=broken)

    assert closes.empty and volumes.empty
    assert isinstance(closes.index, pd.DatetimeIndex)
    assert failed == ["AAA", "BBB"]

def test_select_for_table_renames_the_wanted_coins_and_skips_missing_ones():
    idx = pd.to_datetime(["2023-11-14", "2023-11-15"])
    closes = pd.DataFrame({"BTCUSDT": [10.0, 11.0], "SOLUSDT": [1.0, 2.0],
                           "ETHUSDT": [5.0, 6.0]}, index=idx)

    out = crypto.select_for_table(closes, {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "XRPUSDT": "XRP"})

    assert list(out.columns) == ["BTC", "ETH"]
    assert out["ETH"].tolist() == [5.0, 6.0]
    assert "SOLUSDT" not in out.columns