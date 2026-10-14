import pandas as pd
import pytest
import math
import numpy as np
from momentum.signals import ewma_vol, trend_score,combine_lookbacks,winsorize,loo_market,rolling_beta,residual_returns,residual_trend_score,donchian_position,efficiency_ratio,cross_sectional_zscore



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


def test_residual_returns_use_yesterdays_beta():
    ret = pd.DataFrame({"X": [0.01, 0.08]})
    mkt = pd.DataFrame({"X": [0.01, 0.03]})
    beta = pd.DataFrame({"X": [2.0, 99.0]})    # 99 is poison: today's beta must NOT be used

    res = residual_returns(ret, mkt, beta)

    assert np.isnan(res["X"].iloc[0])           # no yesterday for the first row
    assert res["X"].iloc[1] == pytest.approx(0.02)   # 8% minus 2.0 x 3%


def test_residual_score_finds_the_coin_with_its_own_trend():
    rng = np.random.default_rng(0)
    n = 300
    factor = rng.normal(0.003, 0.02, n)               # the common market move

    def noise():
        return rng.normal(0, 0.004, n)

    ret = pd.DataFrame({
        "A": 1.0 * factor + noise(),
        "B": 1.6 * factor + noise(),
        "C": 0.5 * factor + 0.004 + noise(),          # C has its OWN drift of +0.4% a day
        "D": 1.0 * factor + noise(),
    })
    close = 100 * (1 + ret).cumprod()

    score = residual_trend_score(close, lookback=30, beta_window=60, vol_halflife=30)

    assert score.iloc[-1].idxmax() == "C"           # which coin should the residual score pick?
    assert score.iloc[:60].isna().all().all()         # not enough history for beta yet


def test_donchian_position_places_close_inside_its_range():
    close = pd.DataFrame({"X": [100.0, 104.0, 98.0, 102.0, 106.0, 102.0]})

    pos = donchian_position(close, window=5)

    assert pos["X"].iloc[:4].isna().all()                   # window of 5 needs 5 closes
    assert pos["X"].iloc[4] == pytest.approx(1)           # 106 is the highest of the 5
    assert pos["X"].iloc[5] == pytest.approx(0.5)           # window is 104, 98, 102, 106, 102

def test_efficiency_ratio():
    close = pd.DataFrame({
        "X": [100.0, 102.0, 101.0, 105.0, 107.0],   # net 7, path 2+1+4+2 = 9
        "Z": [100.0, 101.0, 100.0, 101.0, 100.0],   # zigzag, back to start
        "S": [100.0, 101.0, 102.0, 103.0, 104.0],   # straight line
        "F": [5.0] * 5,                             # flat
    })
    er = efficiency_ratio(close, window=4)
    assert er["X"].iloc[:4].isna().all()
    assert er["X"].iloc[4] == pytest.approx(7 / 9)
    assert er["Z"].iloc[4] == pytest.approx(0)
    assert er["S"].iloc[4] == pytest.approx(1)
    assert np.isnan(er["F"].iloc[4])                # flat: 0/0 -> NaN 


def test_cross_sectional_zscore():
    score = pd.DataFrame({
        "A": [1.0, 5.0, np.nan],
        "B": [2.0, 5.0, 2.0],
        "C": [3.0, 5.0, 4.0],
        "D": [6.0, 5.0, 6.0],
    })
    z = cross_sectional_zscore(score)
    std0 = np.sqrt(14 / 3)                       # row 0 std, dividing by n-1
    assert z.loc[0, "A"] == pytest.approx(-2 / std0)
    assert z.loc[0, "C"] == pytest.approx(0 / std0)
    assert z.loc[0, "D"] == pytest.approx(3 / std0)
    assert z.loc[1].isna().all()                 # all equal: 0/0 -> NaN
    assert np.isnan(z.loc[2, "A"])               # missing stays missing
    assert z.loc[2, "B"] == pytest.approx(-1)
    assert z.loc[2, "D"] == pytest.approx(1 )
    