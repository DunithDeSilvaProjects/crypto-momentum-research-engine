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


def test_snapshot_table_one_row_per_series():
    idx = pd.to_datetime(["2026-10-12", "2026-10-13", "2026-10-14"])
    df = pd.DataFrame({"US10": [4.00, np.nan, 3.97],
                       "SPX": [100.0, 110.0, 99.0],
                       "WTI": [70.0, 71.0, np.nan]}, index=idx)
    units = {"US10": "bp", "SPX": "pct", "WTI": "pct"}

    t = snapshot.snapshot_table(df, units, windows=(1, 2))

    assert list(t.index) == ["US10", "SPX", "WTI"]
    assert list(t.columns) == ["last", "last_date", "age_days", "chg_1", "chg_2"]
    assert t.loc["WTI", "last"] == pytest.approx(71.0)
    assert t.loc["WTI", "last_date"] == pd.Timestamp("2026-10-13")
    assert t.loc["WTI", "age_days"] == 1        # table date is the newest row, 14 Oct
    assert t.loc["US10", "chg_1"] == pytest.approx(-3)
    assert np.isnan(t.loc["US10", "chg_2"])
    assert t.loc["SPX", "chg_2"] == pytest.approx(99.0 / 100.0 - 1)