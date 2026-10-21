import pandas as pd

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def parse_klines(rows: list, now: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df[df["close_time"] <= now.value // 10**6]  # drop the last row if it's still running
    out = pd.DataFrame({"close": pd.to_numeric(df["close"]),
                        "quote_volume": pd.to_numeric(df["quote_volume"])})
    out.index = pd.DatetimeIndex(pd.to_datetime(df["open_time"], unit="ms"))
    return out