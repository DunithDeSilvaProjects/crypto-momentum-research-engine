import numpy as np
import pandas as pd
from momentum.snapshot import obs_change


def signal(change: float, normal: float, up_is_good: bool) -> float:
    if pd.isna(change) or pd.isna(normal) or normal == 0:
        return np.nan
    s = np.clip(change / normal, -2, 2) / 2
    return s if up_is_good else -s

def change_series(series: pd.Series, n: int, unit: str) -> pd.Series:
    s = series.dropna()
    if unit == "bp":
        return (s - s.shift(n)) * 100
    if unit == "pct":
        return s / s.shift(n) - 1
    raise ValueError(f"unit must be 'bp' or 'pct', got {unit!r}")

def normal_size(series: pd.Series, n: int, unit: str, lookback: int) -> float:
    ch = change_series(series, n, unit).dropna()
    past = ch.iloc[:-1].tail(lookback)
    if past.empty:
        return np.nan
    return past.abs().mean()

def component_signal(series: pd.Series, n: int, unit: str, up_is_good: bool, lookback: int) -> float:
    change = obs_change(series, n, unit)
    normal = normal_size(series, n, unit, lookback)
    return signal(change, normal, up_is_good)