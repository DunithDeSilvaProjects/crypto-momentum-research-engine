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


def contributions(signals: dict, weights: dict) -> pd.Series:
    return pd.Series({
        k: weights[k] * (0 if pd.isna(signals[k]) else signals[k])
        for k in weights
    })

def regime_label(score: float) -> str:
    if pd.isna(score):
        return "unavailable"
    if score <= -60:
        return "strongly risk-off"
    if score <= -20:
        return "cautious"
    if score < 20:
        return "neutral"
    if score < 60:
        return "constructive"
    return "strongly risk-on"

def component_signals(df: pd.DataFrame, spec: dict, n: int, lookback: int) -> dict:
    out = {}
    for name, c in spec.items():
        if c["col"] not in df.columns:
            out[name] = np.nan
            continue
        out[name] = component_signal(df[c["col"]], n, c["unit"], c["up_is_good"], lookback)
    return out