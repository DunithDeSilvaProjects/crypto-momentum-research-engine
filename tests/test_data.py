import numpy as np
import pandas as pd
import pytest

from momentum.data import Panel


def make_panel():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.DataFrame({
        "BTC": [100, 110, 105, 115, 120],
        "ETH": [200, 210, 205, 215, 220],
    }, index=idx)
    quote_volume = pd.DataFrame({
        "BTC": [1000, 1100, 1050, 1150, 1200],
        "ETH": [2000, 2100, 2050, 2150, 2200],
    }, index=idx)
    funding = pd.DataFrame({
        "BTC": [0.01, 0.02, 0.015, 0.025, 0.03],
        "ETH": [0.02, 0.03, 0.025, 0.035, 0.04],
    }, index=idx)
    return Panel(close=close, quote_volume=quote_volume, funding=funding)


def test_ret_matches_hand_calculation():
    panel = make_panel()
    r = panel.ret["BTC"]
    assert np.isnan(r.iloc[0])                       # day 1 has no previous day
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(105 / 110 - 1)
    assert r.iloc[3] == pytest.approx(115 / 105 - 1)
    assert r.iloc[4] == pytest.approx(120 / 115 - 1)


def test_log_returns_add_up():
    panel = make_panel()
    total = panel.logret["BTC"].iloc[1:].sum()
    assert total == pytest.approx(np.log(120 / 100))  # about 0.18232