import numpy as np
import pytest

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