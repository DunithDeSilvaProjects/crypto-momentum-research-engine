import numpy as np
import pandas as pd


def direction(change: float, dead: float) -> float:
    if pd.isna(change):
        return np.nan
    if abs(change) < dead:
        return 0
    return 1 if change > 0 else -1

def yield_regime(real: float, breakeven: float, equities: float) -> str:
    if any(pd.isna(x) for x in (real, breakeven, equities)):
        return "unavailable"
    if breakeven > 0 and equities < 0:
        return "inflationary risk-off"
    if real < 0 and breakeven <= 0 and equities < 0:
        return "growth scare"
    if real > 0 and breakeven >= 0 and equities > 0:
        return "growth-positive"
    if real < 0 and breakeven >= 0 and equities > 0:
        return "liquidity-positive"
    return "mixed"