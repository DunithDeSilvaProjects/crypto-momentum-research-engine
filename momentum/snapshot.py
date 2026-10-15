import numpy as np
import pandas as pd


def obs_change(series: pd.Series, n: int, unit: str) -> float:
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    last, past = s.iloc[-1], s.iloc[-n-1]
    if unit == "bp":
        return (last - past) * 100
    if unit == "pct":
        return last / past - 1
    raise ValueError(f"unit must be 'bp' or 'pct', got {unit!r}")

def snapshot_table(df: pd.DataFrame, units: dict, windows=(1, 5, 20)) -> pd.DataFrame:
    asof = df.index.max()
    rows = {}
    for col in df.columns:
        s = df[col].dropna()
        row = {
            "last": s.iloc[-1],
            "last_date": s.index[-1],
            "age_days": (asof - s.index[-1]).days,
        }
        for n in windows:
            row[f"chg_{n}"] = obs_change(s, n, units[col])
        rows[col] = row
    return pd.DataFrame.from_dict(rows, orient="index")