import numpy as np
import pandas as pd


def ewma_vol(ret: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially weighted daily volatility: sqrt of the EWMA of squared returns."""
    return np.sqrt(ret.pow(2).ewm(halflife=halflife).mean())

def trend_score(close: pd.DataFrame, vol: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Vol-scaled momentum: L-day log return divided by (daily vol x sqrt(L))."""
    log_ret = np.log(close).diff(lookback)
    return log_ret / (vol * np.sqrt(lookback))

def combine_lookbacks(close: pd.DataFrame, vol: pd.DataFrame, lookbacks: tuple) -> pd.DataFrame:
    """Average of trend_score over several lookbacks. NaN until the longest lookback has data."""
    scores = [trend_score(close, vol, L) for L in lookbacks]
    return sum(scores) / len(scores)

def winsorize(score: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Cap scores at +/- limit. Keeps sign, keeps the coin, leaves NaN alone."""
    return score.clip(lower=-limit, upper=limit)

def loo_market(ret: pd.DataFrame) -> pd.DataFrame:
    """For each coin and day: average return of all OTHER coins."""
    out = {}
    for coin in ret.columns:
        others = ret.drop(columns=coin)    # the table without this coin
        out[coin] = others.mean(axis=1)    # average of the rest, one number per day
    return pd.DataFrame(out)