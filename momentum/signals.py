import numpy as np
import pandas as pd


def ewma_vol(ret: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially weighted daily volatility: sqrt of the EWMA of squared returns."""
    return np.sqrt(ret.pow(2).ewm(halflife=halflife).mean())