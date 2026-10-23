import pandas as pd
import requests
import numpy as np
from momentum.config import SignalConfig
from momentum.signals import combine_lookbacks, ewma_vol

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
