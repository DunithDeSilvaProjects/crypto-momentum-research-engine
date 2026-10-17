import numpy as np
import pytest
import pandas as pd

from momentum import scoring


def test_signal_scales_clips_and_flips():
    assert scoring.signal(6.0, 3.0, up_is_good=True) == pytest.approx(1.0)
    assert scoring.signal(12.0, 3.0, up_is_good=True) == pytest.approx(1.0)
    assert scoring.signal(-1.5, 3.0, up_is_good=True) == pytest.approx(-0.25)
    assert scoring.signal(6.0, 3.0, up_is_good=False) == pytest.approx(-1.0)

def test_signal_is_nan_when_it_cannot_be_computed():
    assert np.isnan(scoring.signal(6.0, 0.0, up_is_good=True))
    assert np.isnan(scoring.signal(np.nan, 3.0, up_is_good=True))

def test_change_series_gives_a_change_at_every_valid_date():
    dgs10 = pd.Series([5.28, 5.22, 5.24, np.nan, 5.21],
                      index=pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09",
                                            "2026-10-12", "2026-10-13"]))
    ch = scoring.change_series(dgs10, 1, "bp")

    assert list(ch.index) == list(pd.to_datetime(["2026-10-07", "2026-10-08",
                                                  "2026-10-09", "2026-10-13"]))
    assert np.isnan(ch.iloc[0])
    assert ch.iloc[1] == pytest.approx(-6)
    assert ch.iloc[2] == pytest.approx(2)
    assert ch.iloc[3] == pytest.approx(-3)

def test_normal_size_is_mean_abs_of_past_moves_excluding_today():
    s = pd.Series([4.00, 4.02, 3.99, 4.03, 4.00, 3.97])
    assert scoring.normal_size(s, 1, "bp", lookback=4) == pytest.approx(3.0)
    assert scoring.normal_size(s, 1, "bp", lookback=2) == pytest.approx(3.5)
    assert np.isnan(scoring.normal_size(pd.Series([4.00, 4.02]), 1, "bp", lookback=4))

def test_component_signal_glues_change_normal_size_and_direction():
    s = pd.Series([4.00, 4.02, 3.99, 4.03, 4.00, 3.97])
    assert scoring.component_signal(s, 1, "bp", True, 4) == pytest.approx(-0.5)
    assert scoring.component_signal(s, 1, "bp", False, 4) == pytest.approx(0.5)

    big = pd.Series([4.00, 4.02, 3.99, 4.03, 4.00, 4.06])
    assert scoring.component_signal(big, 1, "bp", True, 4) == pytest.approx(1.0)

    assert np.isnan(scoring.component_signal(pd.Series([4.00, 4.02]), 1, "bp", True, 4))

def test_contributions_are_weight_times_signal_with_missing_as_zero():
    weights = {"equities": 20, "vix": 15, "spreads": 15, "dollar": 15,
               "real_yield": 20, "curve": 15}
    signals = {"equities": 0.5, "vix": -1.0, "spreads": 0.2, "dollar": np.nan,
               "real_yield": -0.8, "curve": 0.1}

    c = scoring.contributions(signals, weights)

    assert list(c.index) == list(weights)
    assert c["equities"] == pytest.approx(10.0)
    assert c["vix"] == pytest.approx(-15.0)
    assert c["dollar"] == pytest.approx(0.0)
    assert c["real_yield"] == pytest.approx(-16.0)
    assert c.sum() == pytest.approx(-16.5)

def test_regime_label_boundaries():
    assert scoring.regime_label(-60) == "strongly risk-off"
    assert scoring.regime_label(-59.9) == "cautious"
    assert scoring.regime_label(-20) == "cautious"
    assert scoring.regime_label(-19.9) == "neutral"
    assert scoring.regime_label(19.9) == "neutral"
    assert scoring.regime_label(20) == "constructive"
    assert scoring.regime_label(59.9) == "constructive"
    assert scoring.regime_label(60) == "strongly risk-on"
    assert scoring.regime_label(-16.5) == "neutral"
    assert scoring.regime_label(float("nan")) == "unavailable"

def test_component_signals_applies_each_spec_and_survives_a_missing_column():
    df = pd.DataFrame({"US10": [4.00, 4.02, 3.99, 4.03, 4.00, 3.97],
                       "SPX": [100, 110, 99, 108.9, 98.01, 107.811]})
    spec = {
        "rates":  {"col": "US10", "unit": "bp",  "up_is_good": False, "weight": 50},
        "stocks": {"col": "SPX",  "unit": "pct", "up_is_good": True,  "weight": 30},
        "dollar": {"col": "NOPE", "unit": "pct", "up_is_good": False, "weight": 20},
    }

    sig = scoring.component_signals(df, spec, n=1, lookback=4)

    assert list(sig) == ["rates", "stocks", "dollar"]
    assert sig["rates"] == pytest.approx(0.5)
    assert sig["stocks"] == pytest.approx(0.5)
    assert np.isnan(sig["dollar"])