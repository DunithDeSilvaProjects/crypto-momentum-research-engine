import pandas as pd
import numpy as np
from momentum import crypto
import pytest
from momentum.config import SignalConfig


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

def test_trend_signal_is_near_plus_one_for_a_steady_climb_clipped_for_a_fall_and_nan_when_flat():
    cfg = SignalConfig(lookbacks=(4, 16))
    days = np.arange(30)
    up = pd.Series(100 * 1.01 ** days)
    down = pd.Series(100 * 0.99 ** days)
    flat = pd.Series([100.0] * 30)

    assert crypto.trend_signal(up, cfg) == pytest.approx(0.995033, abs=1e-4)
    assert crypto.trend_signal(down, cfg) == pytest.approx(-1.0)    
    assert np.isnan(crypto.trend_signal(flat, cfg))
    
def test_breadth_signal_is_the_share_of_rising_coins_scaled_to_minus_one_plus_one():
    cfg = SignalConfig(lookbacks=(4, 16))
    days = np.arange(30)
    closes = pd.DataFrame({
        "A": 100 * 1.01 ** days,
        "B": 100 * 1.02 ** days,
        "C": 100 * 0.99 ** days,
        "D": 100.0,
    })

    assert crypto.breadth_signal(closes, cfg) == pytest.approx(1/3)
    assert crypto.breadth_signal(closes[["A", "B"]], cfg) == pytest.approx(1)
    assert np.isnan(crypto.breadth_signal(closes[["D"]], cfg))

def test_volume_signal_follows_the_days_direction_scaled_by_how_heavy_the_volume_was():
    quiet_volume = [100.0] * 24

    def sig(last_close, last_volume):
        close = pd.Series([100.0] * 24 + [last_close])
        volume = pd.Series(quiet_volume + [last_volume])
        return crypto.volume_signal(close, volume)

    assert sig(101.0, 200.0) == pytest.approx(np.log(2))
    assert sig(99.0, 200.0) == pytest.approx(-np.log(2))
    assert sig(101.0, 50.0) == pytest.approx(0)
    assert sig(101.0, 1000.0) == pytest.approx(1)
    assert sig(100.0, 200.0) == pytest.approx(0)
    assert np.isnan(crypto.volume_signal(pd.Series([100.0] * 10), pd.Series([100.0] * 10)))

def test_volume_signal_baseline_is_exactly_the_20_days_before_today():
    volume = pd.Series([1000.0] * 4 + [500.0] + [100.0] * 19 + [240.0])
    close = pd.Series([100.0] * 24 + [101.0])

    assert crypto.volume_signal(close, volume) == pytest.approx(np.log(240 /120 ))

def okx_funding_payload():
    rows = [
        ("1792771200000", "0.0000685721961385"),
        ("1792742400000", "0.0000753334163283"),
        ("1792713600000", "0.0000668879046611"),
        ("1792684800000", "0.0000781879787428"),
    ]
    data = [
        {"instId": "BTC-USDT-SWAP", "fundingRate": rate, "realizedRate": rate, "fundingTime": ts}
        for ts, rate in rows
    ]
    return {"code": "0", "msg": "", "data": data}


def test_parse_funding_returns_rates_oldest_first_indexed_by_utc_time():
    rates = crypto.parse_funding(okx_funding_payload())

    assert list(rates.index) == [
        #every 8 hours
        pd.Timestamp("2026-10-22 16:00"),
        pd.Timestamp("2026-10-23 00:00"),
        pd.Timestamp("2026-10-23 08:00"),
        pd.Timestamp("2026-10-23 16:00"),
    ]
    assert rates.iloc[0] == pytest.approx(0.0000781879787428)
    assert rates.iloc[-1] == pytest.approx(0.0000685721961385)


def test_parse_funding_raises_when_okx_reports_an_error():
    with pytest.raises(ValueError):
        crypto.parse_funding({"code": "51001", "msg": "Instrument ID does not exist", "data": []})

def test_daily_funding_sums_each_days_three_payments_and_drops_partial_days():
    idx = pd.date_range("2026-10-20 00:00", periods=7, freq="8h")
    rates = pd.Series([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007], index=idx)

    daily = crypto.daily_funding(rates)

    assert list(daily.index) == [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")]
    assert daily.iloc[0] == pytest.approx(0.0006)
    assert daily.iloc[1] == pytest.approx(0.0015)


def test_daily_funding_on_the_real_okx_sample_keeps_only_the_complete_day():
    daily = crypto.daily_funding(crypto.parse_funding(okx_funding_payload()))

    assert len(daily) == 1
    assert daily.index[0] == pd.Timestamp("2026-10-23")
    assert daily.iloc[0] == pytest.approx(0.00021079, abs=1e-7)

def okx_oi_payload():
    return {"code": "0", "msg": "", "data": [
        ["1792771200000", "3421280245.0902", "5066679909.1656"],
        ["1792684800000", "3365594549.8082", "8295449131.0498"],
        ["1792598400000", "3533971225.8776", "6849134918.3412"],
        ["1792512000000", "3189769410.5172", "6876578377.2892"],
    ]}


def test_parse_open_interest_drops_the_unfinished_bar_and_returns_oldest_first():
    morning = pd.Timestamp("2026-10-24 06:30", tz="UTC")
    evening = pd.Timestamp("2026-10-24 17:00", tz="UTC")

    oi = crypto.parse_open_interest(okx_oi_payload(), morning)
    assert len(oi) == 3
    assert oi.index[0] == pd.Timestamp("2026-10-20 16:00")
    assert oi.iloc[0] == pytest.approx(3189769410.5172)
    assert oi.iloc[-1] == pytest.approx(3365594549.8082)

    assert len(crypto.parse_open_interest(okx_oi_payload(), evening)) == 4


def test_parse_open_interest_raises_when_okx_reports_an_error():
    with pytest.raises(ValueError):
        crypto.parse_open_interest({"code": "51001", "msg": "bad request", "data": []}, pd.Timestamp.now(tz="UTC"))