import numpy as np
import pandas as pd


def ewma_vol(ret: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially weighted daily volatility: sqrt of the EWMA of squared returns."""
    return np.sqrt(ret.pow(2).ewm(halflife=halflife).mean())

def trend_score(close: pd.DataFrame, vol: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Vol-scaled momentum: L-day log return divided by (daily vol x sqrt(L))."""
    log_ret = np.log(close).diff(lookback)
    return log_ret / (vol * np.sqrt(lookback))