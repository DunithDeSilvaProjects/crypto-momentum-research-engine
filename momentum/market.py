import pandas as pd
import yfinance as yf


def fetch_yf_closes(symbols: dict, start: str) -> pd.DataFrame:
    raw = yf.download(list(symbols.values()), start=start, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {list(symbols.values())}")
    closes = raw["Close"]
    return closes.rename(columns={ticker: name for name, ticker in symbols.items()})