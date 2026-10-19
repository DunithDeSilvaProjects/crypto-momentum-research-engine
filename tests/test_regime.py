import numpy as np
import pytest
import pandas as pd 
from momentum import regime


def test_direction_up_down_or_flat_with_a_dead_zone():
    assert regime.direction(14.0, dead=2.0) == 1
    assert regime.direction(-14.0, dead=2.0) == -1
    assert regime.direction(1.5, dead=2.0) == 0
    assert regime.direction(-1.5, dead=2.0) == 0
    assert regime.direction(2.0, dead=2.0) == 1      # exactly on the edge
    assert np.isnan(regime.direction(np.nan, dead=2.0))

def test_yield_regime_names_each_regime_and_falls_back_to_mixed():
    # (real yield, breakeven, equities) -> regime
    assert regime.yield_regime(+1, +1, +1) == "growth-positive"
    assert regime.yield_regime(+1, 0, +1) == "growth-positive"
    assert regime.yield_regime(-1, +1, -1) == "inflationary risk-off"
    assert regime.yield_regime(+1, +1, -1) == "inflationary risk-off"   # real yield can be anything
    assert regime.yield_regime(-1, -1, -1) == "growth scare"
    assert regime.yield_regime(-1, 0, -1) == "growth scare"             # the live reading
    assert regime.yield_regime(-1, +1, +1) == "liquidity-positive"
    assert regime.yield_regime(-1, 0, +1) == "liquidity-positive"
    assert regime.yield_regime(+1, -1, +1) == "mixed"                   # fits no row
    assert regime.yield_regime(0, 0, 0) == "mixed"                      # flat is known, not missing
    assert regime.yield_regime(np.nan, 0, -1) == "unavailable"

def test_yield_regime_report_chains_changes_directions_and_regime():
    df = pd.DataFrame({
        "DFII10": [2.00, 1.98, 1.95, 1.92, 1.90, 1.86],
        "T10YIE": [2.30, 2.31, 2.29, 2.30, 2.31, 2.30],
        "SP500":  [7000, 6990, 6950, 6900, 6880, 6863.5],
    })

    r = regime.yield_regime_report(df)

    assert r["regime"] == "growth scare"
    assert r["changes"]["real_yield"] == pytest.approx(-14)
    assert r["changes"]["breakeven"] == pytest.approx(0)
    assert r["changes"]["equities"] == pytest.approx(-0.0195)
    assert r["directions"] == {"real_yield": -1, "breakeven": 0, "equities": -1}

def test_yield_regime_report_is_unavailable_when_an_input_is_missing():
    df = pd.DataFrame({
        "DFII10": [2.00, 1.98, 1.95, 1.92, 1.90, 1.86],
        "SP500":  [7000, 6990, 6950, 6900, 6880, 6863.5],
    })

    r = regime.yield_regime_report(df)

    assert r["regime"] == "unavailable"
    assert np.isnan(r["changes"]["breakeven"])

def test_render_yield_regime_prints_the_regime_and_each_input():
    rep = {"regime": "growth scare",
           "changes": {"real_yield": -14.0, "breakeven": 0.0, "equities": -0.0195},
           "directions": {"real_yield": -1, "breakeven": 0, "equities": -1}}

    lines = regime.render_yield_regime(rep).split("\n")

    assert lines[0] == "Yield regime (draft rules, to be confirmed): growth scare"
    assert lines[1] == "10y real yield -14.0 bp (down), 10y breakeven +0.0 bp (flat), S&P 500 -1.95% (down)"


def test_render_yield_regime_marks_a_missing_input_as_unknown():
    rep = {"regime": "unavailable",
           "changes": {"real_yield": -14.0, "breakeven": np.nan, "equities": -0.0195},
           "directions": {"real_yield": -1, "breakeven": np.nan, "equities": -1}}

    text = regime.render_yield_regime(rep)

    assert "10y breakeven n/a (unknown)" in text