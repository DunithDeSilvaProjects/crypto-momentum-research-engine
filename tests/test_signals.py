import pandas as pd
import pytest
import math
import numpy as np
from momentum.signals import ewma_vol, trend_score,combine_lookbacks,winsorize,loo_market,rolling_beta



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


def test_combine_lookbacks_averages_and_waits_for_longest_window():
    close = pd.DataFrame({"BTC": [100.0, 101.0, 103.0, 102.0, 106.0, 105.0]})
    vol = pd.DataFrame({"BTC": [0.02] * 6})

    out = combine_lookbacks(close, vol, lookbacks=(2, 4))

    s2 = trend_score(close, vol, 2)
    s4 = trend_score(close, vol, 4)
    assert out["BTC"].iloc[:4].isna().all()                 # the 4-day score needs 4 prior closes
    assert out["BTC"].iloc[4] == pytest.approx((s2["BTC"].iloc[4] + s4["BTC"].iloc[4]) / 2)


def test_winsorize_caps_extremes_and_keeps_nan():
    score = pd.DataFrame({"X": [0.4, -1.1, 0.9, 7.5, -4.2, np.nan]})

    out = winsorize(score, limit=3.0)

    assert out["X"].iloc[:3].tolist() == [0.4, -1.1, 0.9]    # inside the limit: untouched
    assert out["X"].iloc[3] == pytest.approx(3.0)                            # 7.5 capped
    assert out["X"].iloc[4] == pytest.approx(-3.0)                            # -4.2 capped
    assert np.isnan(out["X"].iloc[5])                         # NaN stays NaN


def test_loo_market_excludes_own_return():
    ret = pd.DataFrame({"BTC": [0.04], "ETH": [0.05], "SOL": [0.09]})

    m = loo_market(ret)

    assert m["SOL"].iloc[0] == pytest.approx(0.045)
    assert m["BTC"].iloc[0] == pytest.approx(0.07)
    assert m["ETH"].iloc[0] == pytest.approx(0.065)


def test_rolling_beta_recovers_known_betas():
    mkt = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.01, 0.02, 0.03]})
    ret = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.005, 0.01, 0.015]})

    beta = rolling_beta(ret, mkt, window=3)

    assert beta["A"].iloc[:2].isna().all()
    assert beta["A"].iloc[2] == pytest.approx(2)    # A moves 2% for each 1% of market
    assert beta["B"].iloc[2] == pytest.approx(0.5)    # B moves 0.5% for each 1% of market
    