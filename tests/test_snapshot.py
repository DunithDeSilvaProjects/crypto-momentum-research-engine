import numpy as np
import pandas as pd
import pytest

from momentum import snapshot


def test_obs_change_counts_valid_observations_not_rows():
    dgs10 = pd.Series([5.28, 5.22, 5.24, np.nan, 5.21],
                      index=pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09",
                                            "2026-10-12", "2026-10-13"]))
    assert snapshot.obs_change(dgs10, 1, "bp") == pytest.approx(-3)    # 5.21 vs 9 Oct, in basis points
    assert snapshot.obs_change(dgs10, 3, "bp") == pytest.approx(-7)    # 5.21 vs 7 Oct, in basis points

    sp = pd.Series([7805.45, 7718.94], index=pd.to_datetime(["2026-10-13", "2026-10-14"]))
    assert snapshot.obs_change(sp, 1, "pct") == pytest.approx(7718.94 / 7805.45 - 1)      # a fraction, not a percent
    assert np.isnan(snapshot.obs_change(sp, 5, "pct"))                  # not enough history

def test_obs_change_skips_a_nan_newest_row():
    s = pd.Series([5.0, 5.5, np.nan], index=pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14"]))
    assert snapshot.obs_change(s, 1, "bp") == pytest.approx(50)   # last valid is 5.5 vs 5.0


def test_obs_change_rejects_an_unknown_unit():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="unit must be 'bp' or 'pct'"):                   # a word from the error message
        snapshot.obs_change(s, 1, "percent")