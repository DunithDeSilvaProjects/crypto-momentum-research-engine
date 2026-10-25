import pandas as pd
import numpy as np

def forward_returns(close: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    entry = close.shift(-lag)
    exit_ = close.shift(-(lag + horizon))
    return exit_ / entry - 1


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_coins: int = 5) -> pd.Series:
    mask = signal.notna() & fwd.notna()
    signal_ranks = signal.where(mask).rank(axis=1)
    fwd_ranks = fwd.where(mask).rank(axis=1)
    ic = signal_ranks.corrwith(fwd_ranks, axis=1)
    return ic.where(mask.sum(axis=1) >= min_coins)


def ic_summary(ic: pd.Series, step: int = 1) -> dict:
    sample = ic.dropna().iloc[::step]
    n = len(sample)
    mean = sample.mean()
    t = mean / (sample.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": mean, "t": t, "hit_rate": (sample > 0).mean()}