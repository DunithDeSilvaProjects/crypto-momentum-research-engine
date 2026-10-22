import pandas as pd
import requests

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