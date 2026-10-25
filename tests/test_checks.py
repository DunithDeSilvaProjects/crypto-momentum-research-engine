import numpy as np
import pandas as pd
import pytest

from momentum import checks


def make(curve):
    idx = pd.to_datetime(["2026-10-16", "2026-10-19"])
    return pd.DataFrame({"DGS10": [5.09, 5.15], "DGS2": [4.74, 4.77],
                         "T10Y2Y": curve}, index=idx)


def test_identity_check_uses_the_latest_date_where_all_three_exist():
    out = checks.identity_check(make([0.35, np.nan]), "DGS10", "DGS2", "T10Y2Y")

    assert out["asof"] == pd.Timestamp("2026-10-16")
    assert out["gap"] == pytest.approx(0.00, abs=1e-9)
    assert out["ok"] is True


def test_identity_check_flags_a_gap_larger_than_the_tolerance():
    out = checks.identity_check(make([0.40, 0.45]), "DGS10", "DGS2", "T10Y2Y")

    assert out["asof"] == pd.Timestamp("2026-10-19")
    assert out["gap"] == pytest.approx(-0.07, abs=1e-9)
    assert out["ok"] is False


def test_identity_check_is_unknown_when_inputs_are_missing():
    df = make([np.nan, np.nan])

    assert checks.identity_check(df, "DGS10", "DGS2", "T10Y2Y")["ok"] is None
    assert checks.identity_check(df, "DGS10", "DGS2", "NOPE")["ok"] is None


def test_render_checks_says_ok_mismatch_or_not_checked():
    results = [
        ("10y minus 2y = 2s10s", {"asof": pd.Timestamp("2026-10-22"), "gap": 0.0, "ok": True}),
        ("10y minus real = breakeven", {"asof": pd.Timestamp("2026-10-22"), "gap": 0.05, "ok": False}),
        ("third check", {"asof": pd.NaT, "gap": np.nan, "ok": None}),
    ]

    assert checks.render_checks(results).split("\n") == [
        "10y minus 2y = 2s10s: ok (gap +0.0 bp, as of 22 Oct)",
        "10y minus real = breakeven: MISMATCH (gap +5.0 bp, as of 22 Oct)",
        "third check: not checked (data missing)",
    ]