import pandas as pd
import numpy as np
from momentum import crypto
import pytest
from momentum.config import SignalConfig
import requests


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

def test_funding_signal_is_contrarian_around_the_base_rate():
    def sig(*daily):
        return crypto.funding_signal(pd.Series(daily))

    assert sig(0.0003, 0.0006, 0.0009) == pytest.approx(-0.5)
    assert sig(0.0003, 0.0003, 0.0003) == pytest.approx(0)
    assert sig(0.0009, 0.0009, 0.0009) == pytest.approx(-1)
    assert sig(0.0020, 0.0020, 0.0020) == pytest.approx(-1)
    assert sig(-0.0003, -0.0003, -0.0003) == pytest.approx(1)
    assert sig(0.01, 0.0003, 0.0003, 0.0003) == pytest.approx(0)
    assert np.isnan(sig(0.0003, 0.0003))

def test_oi_signal_needs_rising_open_interest_and_takes_the_price_direction():
    def sig(oi_last, close_last):
        oi = pd.Series([100.0] * 5 + [oi_last])
        close = pd.Series([100.0] * 5 + [close_last])
        return crypto.oi_signal(oi, close)

    assert sig(110.0, 105.0) == pytest.approx(1)
    assert sig(110.0, 95.0) == pytest.approx(-1)
    assert sig(105.0, 105.0) == pytest.approx(0.5)
    assert sig(105.0, 95.0) == pytest.approx(-0.5)
    assert sig(95.0, 105.0) == pytest.approx(0)
    assert sig(130.0, 105.0) == pytest.approx(1)
    assert sig(110.0, 100.0) == pytest.approx(0)
    assert np.isnan(crypto.oi_signal(pd.Series([100.0] * 3), pd.Series([100.0] * 3)))

def test_parse_stablecoin_supply_returns_usd_supply_by_date_oldest_first():
    rows = [
        {"date": "1512086400", "totalCirculatingUSD": {"peggedUSD": 300.0}},
        {"date": "1511913600", "totalCirculatingUSD": {"peggedUSD": 100.0}},
        {"date": "1512000000", "totalCirculatingUSD": {"peggedUSD": 200.0}},
    ]

    supply = crypto.parse_stablecoin_supply(rows)

    assert list(supply.index) == [pd.Timestamp("2017-11-29"), pd.Timestamp("2017-11-30"), pd.Timestamp("2017-12-01")]
    assert list(supply.values) == [100, 200.0, 300]

def test_stablecoin_signal_is_the_weekly_supply_growth_scaled_to_minus_one_plus_one():
    def sig(last):
        return crypto.stablecoin_signal(pd.Series([100.0] * 7 + [last]))

    assert sig(100.5) == pytest.approx(0.5)
    assert sig(101.0) == pytest.approx(1)
    assert sig(103.0) == pytest.approx(1)
    assert sig(99.5) == pytest.approx(-0.5)
    assert sig(100.0) == pytest.approx(0)
    assert np.isnan(crypto.stablecoin_signal(pd.Series([100.0] * 5)))

class OkxReply:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise requests.HTTPError("403 Forbidden")

    def json(self):
        return self.payload


def test_fetch_okx_funding_asks_for_the_swap_and_returns_the_json():
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return OkxReply({"code": "0", "data": []})

    payload = crypto.fetch_okx_funding("BTC-USDT-SWAP", limit=30, get=fake_get)

    assert seen["url"] == "https://www.okx.com/api/v5/public/funding-rate-history"
    assert seen["params"] == {"instId": "BTC-USDT-SWAP", "limit": 30}
    assert seen["timeout"] == 30
    assert payload == {"code": "0", "data": []}


def test_fetch_okx_funding_raises_when_the_http_request_is_refused():
    def refused(url, params=None, timeout=None):
        return OkxReply({}, fail=True)

    with pytest.raises(requests.HTTPError):
        crypto.fetch_okx_funding("BTC-USDT-SWAP", get=refused)


def test_fetch_okx_open_interest_asks_for_daily_history_and_returns_the_json():
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return OkxReply({"code": "0", "data": []})

    payload = crypto.fetch_okx_open_interest("BTC", get=fake_get)

    assert seen["url"] == "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume"
    assert seen["params"] == {"ccy": "BTC", "period": "1D"}
    assert seen["timeout"] == 30
    assert payload == {"code": "0", "data": []}


def test_fetch_okx_open_interest_raises_when_the_http_request_is_refused():
    def refused(url, params=None, timeout=None):
        return OkxReply({}, fail=True)

    with pytest.raises(requests.HTTPError):
        crypto.fetch_okx_open_interest("BTC", get=refused)

def test_fetch_stablecoins_asks_defillama_for_the_full_history_and_returns_the_json():
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return OkxReply([])

    rows = crypto.fetch_stablecoins(get=fake_get)

    assert seen["url"] == "https://stablecoins.llama.fi/stablecoincharts/all"
    assert seen["params"] is None
    assert seen["timeout"] == 30
    assert rows == []


def test_fetch_stablecoins_raises_when_the_http_request_is_refused():
    def refused(url, params=None, timeout=None):
        return OkxReply([], fail=True)

    with pytest.raises(requests.HTTPError):
        crypto.fetch_stablecoins(get=refused)


def test_attempt_returns_the_value_when_loading_works_and_a_note_when_it_fails():
    value, note = crypto.attempt("OKX funding", lambda: 42)
    assert value == 42
    assert note is None

    def broken():
        raise requests.ConnectionError("down")

    value, note = crypto.attempt("OKX funding", broken)
    assert value is None
    assert note == "OKX funding data unavailable today (ConnectionError)"

def test_load_derivatives_parses_each_source_and_survives_one_failing():
    now = pd.Timestamp("2026-10-24 06:30", tz="UTC")
    rows = [
        {"date": "1511913600", "totalCirculatingUSD": {"peggedUSD": 100.0}},
        {"date": "1512000000", "totalCirculatingUSD": {"peggedUSD": 200.0}},
    ]

    def down(ccy):
        raise requests.ConnectionError("down")

    data, notes = crypto.load_derivatives(
        now,
        fetch_funding=lambda inst_id: okx_funding_payload(),
        fetch_oi=lambda ccy: okx_oi_payload(),
        fetch_stable=lambda: rows,
    )
    assert len(data["funding"]) == 1
    assert len(data["oi"]) == 3
    assert len(data["stablecoins"]) == 2
    assert notes == []

    data, notes = crypto.load_derivatives(
        now,
        fetch_funding=lambda inst_id: okx_funding_payload(),
        fetch_oi=down,
        fetch_stable=lambda: rows,
    )
    assert data["oi"] is None
    assert len(data["funding"]) == 1
    assert notes == ["OKX open interest data unavailable today (ConnectionError)"]


def test_crypto_weights_follow_the_brief_and_add_up_to_100():
    assert sum(crypto.CRYPTO_WEIGHTS.values()) == 100
    assert crypto.CRYPTO_WEIGHTS["trend"] == 20


def test_crypto_signals_gathers_all_eight_and_marks_missing_sources_nan():
    days = np.arange(30)
    closes = pd.DataFrame({"BTCUSDT": 100 * 1.01 ** days, "ETHUSDT": 100 * 0.99 ** days})
    volumes = pd.DataFrame({"BTCUSDT": [100.0] * 30, "ETHUSDT": [100.0] * 30})
    derivs = {
        "funding": pd.Series([0.0003, 0.0003, 0.0003]),
        "oi": None,
        "stablecoins": pd.Series([100.0] * 7 + [101.0]),
    }

    sig = crypto.crypto_signals(closes, volumes, derivs, SignalConfig(lookbacks=(4, 16)))

    assert list(sig) == list(crypto.CRYPTO_WEIGHTS)
    assert sig["trend"] == pytest.approx(0.995033, abs=1e-4)
    assert sig["breadth"] == pytest.approx(0)
    assert sig["volume"] == pytest.approx(0)
    assert sig["funding"] == pytest.approx(0)
    assert np.isnan(sig["open_interest"])
    assert sig["stablecoins"] == pytest.approx(1)
    assert np.isnan(sig["etf_flows"])
    assert np.isnan(sig["liquidations"])

def test_crypto_report_scores_with_crypto_weights_and_its_own_coverage_floor():
    days = np.arange(30)
    closes = pd.DataFrame({"BTCUSDT": 100 * 1.01 ** days, "ETHUSDT": 100 * 0.99 ** days})
    volumes = pd.DataFrame({"BTCUSDT": [100.0] * 30, "ETHUSDT": [100.0] * 30})
    derivs = {
        "funding": pd.Series([0.0003, 0.0003, 0.0003]),
        "oi": None,
        "stablecoins": pd.Series([100.0] * 7 + [101.0]),
    }
    cfg = SignalConfig(lookbacks=(4, 16))

    rep = crypto.crypto_report(closes, volumes, derivs, cfg)

    assert rep["score"] == pytest.approx(29.90, abs=1e-3)
    assert rep["coverage"] == pytest.approx(0.70)
    assert rep["label"] == "constructive"

    strict = crypto.crypto_report(closes, volumes, derivs, cfg, min_coverage=0.8)
    assert strict["label"] == "insufficient data"


def test_parse_stablecoin_supply_drops_todays_unfinished_row_when_now_is_given():
    rows = [
        {"date": "1511913600", "totalCirculatingUSD": {"peggedUSD": 100.0}},
        {"date": "1512000000", "totalCirculatingUSD": {"peggedUSD": 200.0}},
        {"date": "1512086400", "totalCirculatingUSD": {"peggedUSD": 300.0}},
    ]
    midday = pd.Timestamp("2017-12-01 12:00", tz="UTC")
    next_day = pd.Timestamp("2017-12-02 00:00", tz="UTC")

    assert len(crypto.parse_stablecoin_supply(rows, midday)) == 2
    assert len(crypto.parse_stablecoin_supply(rows, next_day)) == 3
    assert len(crypto.parse_stablecoin_supply(rows)) == 3


def test_parse_binance_funding_sorts_floats_and_daily_sum_drops_partial_day():
    rows = [
        {"symbol": "BTCUSDT", "fundingTime": 1792857600000, "fundingRate": "0.00000960"},
        {"symbol": "BTCUSDT", "fundingTime": 1792886400000, "fundingRate": "0.00009409"},
        {"symbol": "BTCUSDT", "fundingTime": 1792915200000, "fundingRate": "0.00006005"},
        {"symbol": "BTCUSDT", "fundingTime": 1792944000000, "fundingRate": "0.00008027"},
    ]

    rates = crypto.parse_binance_funding(rows)
    daily = crypto.daily_funding(rates)

    assert len(rates) == 4
    assert rates.iloc[0] == pytest.approx(0.0000096)
    assert len(daily) == 1
    assert daily.iloc[0] == pytest.approx(0.00023441)


def test_parse_binance_funding_raises_on_error_dict():
    with pytest.raises(ValueError, match="Binance error -1121"):
        crypto.parse_binance_funding({"code": -1121, "msg": "Invalid symbol."})


def _fake_funding_get(times, calls):
    def fake_get(url, params):
        calls.append(params["startTime"])
        rows = [
            {"symbol": "X", "fundingTime": t, "fundingRate": "0.0001"}
            for t in times
            if t >= params["startTime"]
        ]
        return rows[: params["limit"]]
    return fake_get


def test_fetch_binance_funding_pages_until_a_short_page():
    calls = []
    fake = _fake_funding_get([1000, 2000, 3000, 4000, 5000], calls)

    rates = crypto.fetch_binance_funding("X", start_ms=0, get=fake, limit=2)

    assert len(rates) == 5
    assert calls == [0, 2001, 4001]


def test_fetch_binance_funding_stops_on_an_empty_page_without_duplicates():
    calls = []
    fake = _fake_funding_get([1000, 2000, 3000, 4000], calls)

    rates = crypto.fetch_binance_funding("X", start_ms=0, get=fake, limit=2)

    assert len(rates) == 4
    assert rates.index.is_unique
    assert calls == [0, 2001, 4001]

def test_funding_panel_uses_each_coins_own_schedule_and_zero_fills_gaps():
    a_times = pd.to_datetime([
        "2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:00",
        "2026-01-02 00:00", "2026-01-02 08:00", "2026-01-02 16:00",
    ])
    b_times = pd.to_datetime([
        "2026-01-01 00:00", "2026-01-01 12:00", "2026-01-02 00:00",
        "2026-01-02 12:00", "2026-01-03 00:00",
    ])
    rates = {
        "A": pd.Series([0.0001] * 3 + [0.0002] * 3, index=a_times),
        "B": pd.Series([0.0005, 0.0005, 0.0001, 0.0001, 0.0009], index=b_times),
    }
    index = pd.date_range("2026-01-01", periods=3)

    panel = crypto.funding_panel(["A", "B"], start_ms=0, index=index,
                                 fetch=lambda symbol, start_ms: rates[symbol])

    np.testing.assert_allclose(panel["A"], [0.0003, 0.0006, 0])
    np.testing.assert_allclose(panel["B"], [0.001, 0.0002, 0])
    assert list(panel.columns) == ["A", "B"]
    assert panel.index.equals(index)

def test_funding_panel_matches_a_timezone_aware_index():
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:00"])
    rates = pd.Series([0.0001, 0.0002, 0.0003], index=times)
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")

    panel = crypto.funding_panel(["A"], start_ms=0, index=index,
                                 fetch=lambda symbol, start_ms: rates)

    np.testing.assert_allclose(panel["A"], [0.0006, 0])
    assert panel.index.equals(index)
    