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