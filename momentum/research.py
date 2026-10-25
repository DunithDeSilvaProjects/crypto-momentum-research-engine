import pandas as pd
import numpy as np
from momentum.config import SignalConfig
from momentum.signals import combine_lookbacks, cross_sectional_zscore, ewma_vol, liquidity_mask, winsorize


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


def momentum_signal(
    close: pd.DataFrame,
    quote_volume: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    liq_window: int = 30,
    min_volume: float = 10_000_000.0,
) -> pd.DataFrame:
    vol = ewma_vol(close.pct_change(), cfg.vol_halflife)
    raw = combine_lookbacks(close, vol, cfg.lookbacks)
    capped = winsorize(raw, cfg.winsor)
    liquid = liquidity_mask(quote_volume, liq_window, min_volume)
    return cross_sectional_zscore(capped.where(liquid))