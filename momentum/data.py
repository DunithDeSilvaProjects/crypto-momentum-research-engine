from dataclasses import dataclass
import pandas as pd
import numpy as np
import requests

@dataclass
class Panel:
    close : pd.DataFrame
    quote_volume : pd.DataFrame
    funding: pd.DataFrame

    @property
    def ret(self) -> pd.DataFrame:
        return self.close.pct_change()

    @property
    def logret(self) -> pd.DataFrame:
        return np.log(self.close).diff()


BASE = "https://fapi.binance.com"

def fetch_klines(symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Download recent candles for one symbol; returns completed candles only."""
    r = requests.get(f"{BASE}/fapi/v1/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=30)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    now_ms = pd.Timestamp.now(tz="UTC").value // 10**6
    df = df[df["close_time"] <= now_ms]                      # drop the unfinished candle
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c])
    return df.set_index("open_time")[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]]