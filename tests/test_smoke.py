import pandas as pd
import numpy as np

def test_rolling_mean_by_hand():
    df = pd.Series([1, 2, 3, 4, 5])
    window = 2
    expected = pd.Series([np.nan, 1.5, 2.5, 3.5, 4.5])
    result = df.rolling(window=window).mean()
    pd.testing.assert_series_equal(result, expected)

def test_pct_change_by_hand():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = prices.pct_change()
    expected = pd.Series([ np.nan,  0.1, -0.1])
    pd.testing.assert_series_equal(result, expected)

def test_deliberately_fails():
    assert 1 + 1 == 2