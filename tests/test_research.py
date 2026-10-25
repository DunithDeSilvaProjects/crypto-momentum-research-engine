import numpy as np
import pandas as pd
import pytest
from momentum import research


def test_forward_returns_start_after_the_execution_lag_and_end_in_nan():
    close = pd.DataFrame({"A": [100.0, 200.0, 100.0, 400.0, 100.0]})

    lag1 = research.forward_returns(close, horizon=1, lag=1)["A"].to_numpy()
    lag0 = research.forward_returns(close, horizon=1, lag=0)["A"].to_numpy()

    np.testing.assert_allclose(lag1, [-0.5, 3.0, -0.75, np.nan, np.nan])
    np.testing.assert_allclose(lag0, [1.0, -0.5, 3.0, -0.75, np.nan])


def test_rank_ic_is_the_daily_spearman_correlation_and_needs_enough_coins():
    cols = list("ABCDE")
    signal = pd.DataFrame(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, np.nan]], columns=cols, dtype=float
    )
    fwd = pd.DataFrame([[0.01, 0.02, 0.03, 0.05, 0.04]] * 3, columns=cols)

    strict = research.rank_ic(signal, fwd, min_coins=5)
    loose = research.rank_ic(signal, fwd, min_coins=4)

    assert strict.iloc[0] == pytest.approx(0.9)
    assert strict.iloc[1] == pytest.approx(-0.9)
    assert np.isnan(strict.iloc[2])
    assert loose.iloc[2] == pytest.approx(1)


def test_ic_summary_samples_every_step_th_day_to_avoid_overlap():
    ic = pd.Series([np.nan, 0.1, 0.3, -0.1, 0.5, 0.1, 0.3])

    daily = research.ic_summary(ic, step=1)
    weekly = research.ic_summary(ic, step=2)

    assert daily["n"] == 6
    assert daily["mean"] == pytest.approx(0.2)
    assert daily["t"] == pytest.approx(2.3355, abs=1e-3)
    assert daily["hit_rate"] == pytest.approx(5 / 6)

    assert weekly["n"] == 3
    assert weekly["mean"] == pytest.approx(0.1/3)
    assert weekly["t"] == pytest.approx(0.5)
    assert weekly["hit_rate"] == pytest.approx(2 / 3)