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