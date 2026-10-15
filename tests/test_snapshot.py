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