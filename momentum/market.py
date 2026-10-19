import pandas as pd
import yfinance as yf
import numpy as np


def fetch_yf_closes(symbols: dict, start: str) -> pd.DataFrame:
    raw = yf.download(list(symbols.values()), start=start, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {list(symbols.values())}")
    closes = raw["Close"]
    return closes.rename(columns={ticker: name for name, ticker in symbols.items()})


def last_complete_session(now: pd.Timestamp) -> pd.Timestamp:
    ny = now.tz_convert("America/New_York")
    day = ny.date() if ny.hour >= 17 else ny.date() - pd.Timedelta(days=1) 
    return pd.Timestamp(np.busday_offset(day, 0, roll="backward"))

def drop_incomplete(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return df[df.index <= cutoff]

def render_asof(now: pd.Timestamp, cutoff: pd.Timestamp) -> str:
    def at(zone: str) -> str:
        return now.tz_convert(zone).strftime("%d %b %H:%M")

    head = (f"Generated: {at('UTC')} UTC · {at('Asia/Singapore')} Singapore/Hong Kong"
            f" · {at('America/New_York')} New York")
    return head + "\n" + f"Market data through last complete US session: {cutoff:%a %d %b %Y} (close 16:00 ET)"