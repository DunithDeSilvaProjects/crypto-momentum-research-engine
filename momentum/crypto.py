import pandas as pd
import requests
import numpy as np
from momentum.config import SignalConfig
from momentum.signals import combine_lookbacks, ewma_vol
from momentum import scoring

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def parse_klines(rows: list, now: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df[df["close_time"] <= now.value // 10**6]  # drop the last row if it's still running
    out = pd.DataFrame({"close": pd.to_numeric(df["close"]),
                        "quote_volume": pd.to_numeric(df["quote_volume"])})
    out.index = pd.DatetimeIndex(pd.to_datetime(df["open_time"], unit="ms"))
    return out

SPOT_BASE = "https://data-api.binance.vision"

def fetch_spot_klines(symbol: str, limit: int = 400, get=requests.get) -> list:
    url = f"{SPOT_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": "1d", "limit": limit}
    r = get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def load_spot_panel(symbols, now, fetch=fetch_spot_klines):
    closes, volumes, failed = {}, {}, []
    for symbol in symbols:
        try:
            table = parse_klines(fetch(symbol), now)
        except Exception:
            failed.append(symbol)
            continue
        closes[symbol] = table["close"]
        volumes[symbol] = table["quote_volume"]
    if not closes:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        return empty, empty.copy(), failed
    return pd.DataFrame(closes), pd.DataFrame(volumes), failed

def select_for_table(closes: pd.DataFrame, names: dict) -> pd.DataFrame:
    wanted = [symbol for symbol in names if symbol in closes.columns]
    return closes[wanted].rename(columns=names)

def trend_signal(close: pd.Series, cfg: SignalConfig = SignalConfig()) -> float:
    frame = close.to_frame("x")
    vol = ewma_vol(frame.pct_change(), cfg.vol_halflife)
    score = combine_lookbacks(frame, vol, cfg.lookbacks)["x"].iloc[-1]
    return float(np.clip(score, -cfg.winsor, cfg.winsor) / cfg.winsor)

def breadth_signal(closes: pd.DataFrame, cfg: SignalConfig = SignalConfig()) -> float:
    trends = pd.Series({coin: trend_signal(closes[coin], cfg) for coin in closes.columns}).dropna()
    if trends.empty:
        return float("nan")
    share = (trends > 0).mean()
    return float(2 * share - 1)

def volume_signal(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    if len(close) < window + 2:
        return float("nan")
    base = volume.iloc[-(window + 1):-1].mean()
    intensity = float(np.clip(np.log(volume.iloc[-1] / base), 0, 1))
    direction = np.sign(close.iloc[-1] - close.iloc[-2])
    return float(direction * intensity)

def parse_funding(payload: dict) -> pd.Series:
    if payload.get("code") != "0":
        raise ValueError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
    rows = payload["data"]
    times = pd.to_datetime([int(row["fundingTime"]) for row in rows], unit="ms")
    rates = pd.Series([float(row["fundingRate"]) for row in rows], index=times)
    return rates.sort_index()

def daily_funding(rates: pd.Series, per_day: int = 3) -> pd.Series:
    grouped = rates.groupby(rates.index.normalize())
    total = grouped.sum()
    count = grouped.count()
    return total[count == per_day]

def parse_open_interest(payload: dict, now: pd.Timestamp) -> pd.Series:
    if payload.get("code") != "0":
        raise ValueError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
    now_ms = now.value // 10**6
    day_ms = 86_400_000
    rows = [row for row in payload["data"] if int(row[0]) + day_ms <= now_ms]
    times = pd.to_datetime([int(row[0]) for row in rows], unit="ms")
    oi = pd.Series([float(row[1]) for row in rows], index=times)
    return oi.sort_index()

def funding_signal(daily: pd.Series, window: int = 3, base: float = 0.0003, scale: float = 0.0006) -> float:
    if len(daily) < window:
        return float("nan")
    excess = (daily.iloc[-window:].mean() - base) / scale
    return float(- np.clip(excess, -1, 1))

def oi_signal(oi: pd.Series, close: pd.Series, window: int = 5, scale: float = 0.10) -> float:
    if len(oi) < window + 1 or len(close) < window + 1:
        return float("nan")
    oi_change = oi.iloc[-1] / oi.iloc[-(window + 1)] - 1
    direction = np.sign(close.iloc[-1] - close.iloc[-(window + 1)])
    return float(direction * np.clip(oi_change / scale, 0, 1))

def parse_stablecoin_supply(rows: list) -> pd.Series:
    dates = pd.to_datetime([int(row["date"]) for row in rows], unit="s")
    values = [float(row["totalCirculatingUSD"]["peggedUSD"]) for row in rows]
    return pd.Series(values, index=dates).sort_index()

def stablecoin_signal(supply: pd.Series, window: int = 7, scale: float = 0.01) -> float:
    if len(supply) < window + 1:
        return float("nan")
    growth = supply.iloc[-1] / supply.iloc[-(window + 1)] - 1
    return float(np.clip(growth / scale, -1, 1))


OKX_BASE = "https://www.okx.com"


def fetch_okx_funding(inst_id: str, limit: int = 100, get=requests.get) -> dict:
    response = get(
        f"{OKX_BASE}/api/v5/public/funding-rate-history",
        params={"instId": inst_id, "limit": limit},
        timeout=limit,
    )
    response.raise_for_status()
    return response.json()

def fetch_okx_open_interest(ccy="BTC", get=requests.get):
    response = get(
        f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume",
        params={"ccy": ccy, "period": "1D"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


LLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def fetch_stablecoins(get=requests.get) -> list:
    response = get(LLAMA_STABLECOINS_URL, timeout=30)
    response.raise_for_status()
    return response.json()

def attempt(label: str, load):
    try:
        return load(), None
    except Exception as exc:
        return None, f"{label} data unavailable today ({type(exc).__name__})"


def load_derivatives(
    now: pd.Timestamp,
    fetch_funding=fetch_okx_funding,
    fetch_oi=fetch_okx_open_interest,
    fetch_stable=fetch_stablecoins,
):
    funding, funding_note = attempt(
        "OKX funding", lambda: daily_funding(parse_funding(fetch_funding("BTC-USDT-SWAP")))
    )
    oi, oi_note = attempt("OKX open interest", lambda: parse_open_interest(fetch_oi("BTC"), now))
    stable, stable_note = attempt("DefiLlama stablecoin", lambda: parse_stablecoin_supply(fetch_stable()))
    notes = [note for note in (funding_note, oi_note, stable_note) if note is not None]
    return {"funding": funding, "oi": oi, "stablecoins": stable}, notes


CRYPTO_WEIGHTS = {
    "trend": 20,
    "breadth": 15,
    "volume": 15,
    "funding": 10,
    "open_interest": 10,
    "stablecoins": 10,
    "etf_flows": 10,
    "liquidations": 10,
}


def crypto_signals(closes, volumes, derivs, cfg: SignalConfig = SignalConfig(), coin: str = "BTCUSDT") -> dict:
    nan = float("nan")
    funding, oi, stable = derivs["funding"], derivs["oi"], derivs["stablecoins"]
    return {
        "trend": trend_signal(closes[coin], cfg),
        "breadth": breadth_signal(closes, cfg),
        "volume": volume_signal(closes[coin], volumes[coin]),
        "funding": nan if funding is None else funding_signal(funding),
        "open_interest": nan if oi is None else oi_signal(oi, closes[coin]),
        "stablecoins": nan if stable is None else stablecoin_signal(stable),
        "etf_flows": nan,
        "liquidations": nan,
    }


CRYPTO_MIN_COVERAGE = 0.7


def crypto_report(closes, volumes, derivs, cfg: SignalConfig = SignalConfig(), k: int = 2,
                  min_coverage: float = CRYPTO_MIN_COVERAGE) -> dict:
    signals = crypto_signals(closes, volumes, derivs, cfg)
    return scoring.report_from_signals(signals,CRYPTO_WEIGHTS , k, min_coverage)