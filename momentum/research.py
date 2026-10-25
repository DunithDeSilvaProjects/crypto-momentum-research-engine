import pandas as pd


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