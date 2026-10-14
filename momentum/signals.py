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

def rolling_beta(ret: pd.DataFrame, mkt: pd.DataFrame, window: int) -> pd.DataFrame:
    """For each coin: rolling cov(coin, its market) / rolling var(its market)."""
    out = {}
    for coin in ret.columns:
        cov = ret[coin].rolling(window=window).cov(mkt[coin])
        var = mkt[coin].rolling(window=window).var()
        out[coin] = cov / var
    return pd.DataFrame(out)

def residual_returns(ret: pd.DataFrame, mkt: pd.DataFrame, beta:pd.DataFrame) -> pd.DataFrame:
    """Coin return minus (yesterday's beta x today's market return)."""
    return ret-(beta.shift(1)*mkt)

def residual_trend_score(close: pd.DataFrame, lookback: int, beta_window: int, vol_halflife: float) -> pd.DataFrame:
    """Vol-scaled momentum of what is left of each coin's return after removing the market."""
    ret = close.pct_change()
    mkt = loo_market(ret)                       # each coin's market, leaving itself out
    beta = rolling_beta(ret, mkt, beta_window)    # how strongly each coin follows it
    res = residual_returns(ret, mkt, beta)            # the coin's return with the market part removed
    vol = ewma_vol(res, vol_halflife)         # how wild the residuals are
    return res.rolling(lookback).sum() / (vol * np.sqrt(lookback))

def donchian_position(close:pd.DataFrame,window:int) -> pd.DataFrame:
    """For each coin: (today's close - L-day low) / (L-day high - L-day low)."""
    lo = close.rolling(window=window).min()
    hi = close.rolling(window=window).max()
    return (close - lo) / (hi - lo)

def efficiency_ratio(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Net move over `window` days divided by the total distance travelled: 1 = straight line, 0 = pure zigzag."""
    net = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return net / path

def cross_sectional_zscore(score: pd.DataFrame) -> pd.DataFrame:
    """Standardise each day across coins: mean 0, std 1 within every row."""
    mean = score.mean(axis=1)
    std = score.std(axis=1)
    return score.sub(mean, axis=0).div(std, axis=0)