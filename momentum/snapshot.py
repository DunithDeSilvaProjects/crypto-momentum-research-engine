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