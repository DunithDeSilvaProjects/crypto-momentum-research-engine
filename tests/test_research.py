import numpy as np
import pandas as pd
import pytest
from momentum import research
from momentum.config import SignalConfig


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


def test_momentum_signal_masks_illiquid_coins_and_standardises_each_day():
    days = np.arange(40)
    close = pd.DataFrame({
        "A": 100 * 1.01 ** days, "B": 100 * 1.02 ** days,
        "C": 100 * 0.99 ** days, "D": 100 * 1.05 ** days,
    })
    volume = pd.DataFrame({"A": 5e7, "B": 5e7, "C": 5e7, "D": 1e3}, index=close.index)

    sig = research.momentum_signal(close, volume, SignalConfig(lookbacks=(4,)))
    last = sig.iloc[-1]

    assert np.isnan(last["D"])
    assert last["C"] == last[["A", "B", "C"]].min()
    assert last[["A", "B", "C"]].mean() == pytest.approx(0, abs=1e-9)
    assert last[["A", "B", "C"]].std() == pytest.approx(1)
    assert sig.iloc[:28].isna().all().all()


def test_momentum_signal_never_uses_the_future():
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (200, 5)), axis=0)), columns=list("ABCDE")
    )
    volume = pd.DataFrame(5e7, index=close.index, columns=close.columns)
    cut = 150

    full = research.momentum_signal(close, volume)
    part = research.momentum_signal(close.iloc[: cut + 1], volume.iloc[: cut + 1])

    pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


def test_long_short_weights_are_dollar_neutral_with_gross_one_and_flat_when_too_few_coins():
    cols = list("ABCDEF")
    signal = pd.DataFrame(
        [[3, 1, 2, 6, 5, 4], [3, np.nan, 2, np.nan, np.nan, 4], [3, np.nan, 2, 6, 5, 4]],
        columns=cols, dtype=float,
    )

    w = research.long_short_weights(signal, k=2)

    np.testing.assert_allclose(w.iloc[0], [0, -0.25, -0.25, +0.25, +0.25, 0])
    np.testing.assert_allclose(w.iloc[1], [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(w.iloc[2], [-0.25, 0, -0.25, 0.25, 0.25, 0])
    assert w.iloc[0].sum() == pytest.approx(0)
    assert w.iloc[0].abs().sum() == pytest.approx(1)