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


def long_short_weights(signal: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    valid = signal.notna().sum(axis=1)
    rank_low = signal.rank(axis=1, method="first")
    rank_high = signal.rank(axis=1, method="first", ascending=False)
    long = rank_high <= k
    short = rank_low <= k
    weights = long * (0.5 / k) - short * (0.5 / k)
    return weights.where(valid >= 2 * k, 0.0).astype(float)


def hold_weights(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    kept = weights.iloc[::horizon]
    return kept.reindex(weights.index).ffill()


def portfolio_returns(weights: pd.DataFrame, close: pd.DataFrame, lag: int = 1) -> pd.Series:
    asset_returns = close.pct_change()
    held = weights.shift(lag + 1)
    return (held * asset_returns).sum(axis=1)


def turnover(weights: pd.DataFrame, lag: int = 1) -> pd.Series:
    held = weights.shift(lag + 1).fillna(0.0)
    return held.diff().abs().sum(axis=1)

def net_returns(gross: pd.Series, turn: pd.Series, cost_bps: float = 10.0) -> pd.Series:
    return gross - turn * cost_bps / 10000

def funding_returns(weights: pd.DataFrame, funding: pd.DataFrame, lag: int = 1) -> pd.Series:
    held = weights.shift(lag + 1)
    return - (held * funding).sum(axis=1)

def sharpe(returns: pd.Series, periods: int = 365) -> float:
    std = returns.std(ddof=1)
    if np.isnan(std) or std == 0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods))