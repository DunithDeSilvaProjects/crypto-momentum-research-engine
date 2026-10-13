import pandas as pd
import pytest

from momentum.signals import ewma_vol


def test_ewma_vol_matches_hand_calculation():
    # daily returns +1%, -2%, +3%; halflife=1 gives weights 1, 0.5, 0.25 (newest first)
    ret = pd.DataFrame({"BTC": [0.01, -0.02, 0.03]})

    vol = ewma_vol(ret, halflife=1)
    expected = ((0.03**2 * 1 + 0.02**2 * 0.5 + 0.01**2 * 0.25) / 1.75) ** 0.5


    assert vol["BTC"].iloc[0] == pytest.approx(0.01)       # one observation: sqrt(0.01**2)
    assert vol["BTC"].iloc[-1] == pytest.approx(expected)   # sqrt(0.000643), by hand