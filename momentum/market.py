import pandas as pd
import yfinance as yf
import numpy as np


def fetch_yf_closes(symbols: dict, start: str, download=None) -> pd.DataFrame:
    raw = yf.download(list(symbols.values()), start=start, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {list(symbols.values())}")
    closes = raw["Close"].rename(columns={ticker: name for name, ticker in symbols.items()})
    return refill_empty(closes, symbols, start, download or download_single_close)

def fetch_yf_optional(symbols, start, fetch=fetch_yf_closes):
    try:
        return fetch(symbols, start=start), None
    except Exception as exc:
        return pd.DataFrame(index=pd.DatetimeIndex([])), f"Yahoo Finance data unavailable today ({type(exc).__name__})"

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


def refill_empty(closes: pd.DataFrame, symbols: dict, start: str, download) -> pd.DataFrame:
    out = closes.copy()
    for name, ticker in symbols.items():
        if name in out.columns and out[name].notna().any():
            continue
        single = download(ticker, start)
        if len(single) > 0:
            out[name] = single.reindex(out.index)
    return out


def download_single_close(ticker: str, start: str) -> pd.Series:
    raw = yf.download(ticker, start=start, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"]
    return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close

def prefer_fresh(fresh: pd.Series, fallback: pd.Series) -> pd.Series:
    return fresh.combine_first(fallback)


def overlay_fresh(frame: pd.DataFrame, fresh: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = frame.reindex(frame.index.union(fresh.index))
    for target, source in mapping.items():
        if source in fresh.columns:
            out[target] = prefer_fresh(fresh[source], frame[target]).reindex(out.index)
    return out