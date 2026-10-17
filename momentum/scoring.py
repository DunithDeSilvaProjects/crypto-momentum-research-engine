import numpy as np
import pandas as pd


def signal(change: float, normal: float, up_is_good: bool) -> float:
    if pd.isna(change) or pd.isna(normal) or normal == 0:
        return np.nan
    s = np.clip(change / normal, -2, 2) / 2
    return s if up_is_good else -s