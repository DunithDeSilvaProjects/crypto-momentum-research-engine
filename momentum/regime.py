import numpy as np
import pandas as pd


def direction(change: float, dead: float) -> float:
    if pd.isna(change):
        return np.nan
    if abs(change) < dead:
        return 0
    return 1 if change > 0 else -1