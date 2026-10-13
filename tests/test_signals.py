import pandas as pd
import pytest
import math
from momentum.signals import ewma_vol, trend_score



def test_ewma_vol_matches_hand_calculation():
    # daily returns +1%, -2%, +3%; halflife=1 gives weights 1, 0.5, 0.25 (newest first)
    ret = pd.DataFrame({"BTC": [0.01, -0.02, 0.03]})

    vol = ewma_vol(ret, halflife=1)
    expected = ((0.03**2 * 1 + 0.02**2 * 0.5 + 0.01**2 * 0.25) / 1.75) ** 0.5


    assert vol["BTC"].iloc[0] == pytest.approx(0.01)       # one observation: sqrt(0.01**2)
    assert vol["BTC"].iloc[-1] == pytest.approx(expected)   # sqrt(0.000643), by hand


def test_trend_score_matches_hand_calculation():
    close = pd.DataFrame({"BTC": [100.0, 102.0, 99.0, 105.0]})
    vol = pd.DataFrame({"BTC": [0.02, 0.02, 0.02, 0.02]})   # constant 2% daily vol

    score = trend_score(close, vol, lookback=3)

    assert score["BTC"].iloc[:3].isna().all()               # fewer than 3 days of history
    expected = math.log(105/100) / (0.02 * math.sqrt(3))         # ln(105/100) / (2% x sqrt 3)
    assert score["BTC"].iloc[3] == pytest.approx(expected)
