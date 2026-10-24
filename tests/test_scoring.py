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

def test_coverage_is_the_share_of_weight_that_has_a_signal():
    weights = {"rates": 50, "stocks": 30, "dollar": 20}

    # dollar missing: we see 50 + 30 of 100
    sig = {"rates": 0.5, "stocks": -0.2, "dollar": np.nan}
    assert scoring.coverage(sig, weights) == pytest.approx(0.8)

    # nothing missing
    sig = {"rates": 0.5, "stocks": -0.2, "dollar": 0.1}
    assert scoring.coverage(sig, weights) == pytest.approx(1.0)
    # everything missing
    sig = {"rates": np.nan, "stocks": np.nan, "dollar": np.nan}
    assert scoring.coverage(sig, weights) == pytest.approx(0.0)

    # weights that do not sum to 100: divide by their own total
    assert scoring.coverage({"a": 1.0, "b": np.nan}, {"a": 1, "b": 3}) == pytest.approx(0.25)

def test_top_drivers_picks_the_largest_pushes_each_way():
    c = pd.Series({"equities": 10, "vix": -15, "spreads": 3,
                   "dollar": 0, "real_yield": -16, "curve": 1.5})

    d = scoring.top_drivers(c, k=2)
    assert d["positive"] == [("equities", 10), ("spreads", 3)]
    assert d["negative"] == [("real_yield", -16), ("vix", -15)]

    # k larger than what exists: the zero never appears
    d = scoring.top_drivers(c, k=5)
    assert len(d["positive"]) == 3
    assert len(d["negative"]) == 2

def test_traditional_report_chains_signals_score_coverage_and_drivers():
    df = pd.DataFrame({"US10": [4.00, 4.02, 3.99, 4.03, 4.00, 3.97],
                       "SPX": [100, 110, 99, 108.9, 98.01, 107.811]})
    spec = {
        "rates":  {"col": "US10", "unit": "bp",  "up_is_good": False, "weight": 50},
        "stocks": {"col": "SPX",  "unit": "pct", "up_is_good": True,  "weight": 30},
        "dollar": {"col": "NOPE", "unit": "pct", "up_is_good": False, "weight": 20},
    }

    r = scoring.traditional_report(df, spec, n=1, lookback=4, k=2)

    assert r["score"] == pytest.approx(40)
    assert r["coverage"] == pytest.approx(0.8)
    assert r["label"] == scoring.regime_label(r["score"])
    pos = r["drivers"]["positive"]
    assert [name for name, _ in pos] == ["rates", "stocks"]
    assert [value for _, value in pos] == pytest.approx([25, 15])
    assert r["drivers"]["negative"] == []
    assert list(r["signals"]) == ["rates", "stocks", "dollar"]

def test_render_score_prints_headline_and_drivers():
    report = {"score": -16.5, "label": "neutral", "coverage": 0.85,
              "drivers": {"positive": [("equities", 10.0), ("spreads", 3.0)],
                          "negative": [("real_yield", -16.0), ("vix", -15.0)]}}
    names = {"equities": "Equity trend", "real_yield": "10y real yield"}

    lines = scoring.render_score(report, names).split("\n")

    assert lines[0] == "Traditional regime: -16.5 (neutral)  coverage 85%"
    assert lines[1] == "Pushing up:   Equity trend +10.0, spreads +3.0"
    assert lines[2] == "Pushing down: 10y real yield -16.0, vix -15.0"


def test_render_score_says_none_when_a_side_is_empty():
    report = {"score": 0.0, "label": "neutral", "coverage": 1.0,
              "drivers": {"positive": [], "negative": []}}

    lines = scoring.render_score(report).split("\n")

    assert lines[1] == "Pushing up:   none"
    assert lines[2] == "Pushing down: none"


def test_stale_columns_flags_old_or_empty_series():
    idx = pd.to_datetime(["2026-10-09", "2026-10-12", "2026-10-15", "2026-10-16"])
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],                       # last valid 16 Oct, age 0
        "B": [1, 2, 3, np.nan],                  # last valid 15 Oct, age 1
        "C": [1, np.nan, np.nan, np.nan],        # last valid 09 Oct, age 7
        "D": [np.nan, np.nan, np.nan, np.nan],   # never valid
    }, index=idx)

    assert scoring.stale_columns(df, max_age_days=3) == ["C", "D"]
    # age exactly equal to the limit is allowed
    assert scoring.stale_columns(df, max_age_days=7) == ["D"]


def test_stale_columns_counts_business_days_so_a_weekend_is_not_staleness():
    idx = pd.to_datetime(["2026-10-09", "2026-10-12", "2026-10-15", "2026-10-16", "2026-10-19"])
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],                          # last valid Mon 19 Oct, age 0
        "B": [1, 2, 3, 4, np.nan],                     # last valid Fri 16 Oct, age 1
        "C": [1, 2, 3, np.nan, np.nan],                # last valid Thu 15 Oct, age 2
        "D": [1, np.nan, np.nan, np.nan, np.nan],      # last valid Fri 9 Oct, age 6
        "E": [np.nan] * 5,                             # never valid
    }, index=idx)

    assert scoring.stale_columns(df, max_age_days=3) == ["D", "E"]
    assert scoring.stale_columns(df, max_age_days=2) == ["D", "E"]      # C is exactly at the limit
    assert scoring.stale_columns(df, max_age_days=1) == ["C", "D", "E"]

def test_guarded_label_withholds_the_label_when_coverage_is_too_low():
    assert scoring.guarded_label(-24.0, 1.0, 0.8) == "cautious"
    assert scoring.guarded_label(-15.9, 0.7, 0.8) == "insufficient data"
    assert scoring.guarded_label(-15.9, 0.8, 0.8) == "neutral"
    assert scoring.guarded_label(np.nan, 1.0, 0.8) == "unavailable"

def test_traditional_report_withholds_the_label_when_a_component_is_missing():
    idx = pd.bdate_range("2026-06-01", periods=80)
    t = np.arange(80)
    df = pd.DataFrame({"A": 100 + t + 3 * np.sin(t)}, index=idx)
    spec = {
        "a": {"col": "A", "unit": "pct", "up_is_good": True, "weight": 60},
        "b": {"col": "B", "unit": "pct", "up_is_good": True, "weight": 40},
    }

    low = scoring.traditional_report(df, spec, n=5, lookback=60)
    ok = scoring.traditional_report(df, spec, n=5, lookback=60, min_coverage=0.5)

    assert low["coverage"] == pytest.approx(0.6)
    assert low["label"] == "insufficient data"
    assert ok["label"] != "insufficient data"


def test_report_from_signals_scores_labels_and_explains():
    sig = {"a": 0.5, "b": -1.0, "c": np.nan}
    weights = {"a": 50, "b": 30, "c": 20}

    rep = scoring.report_from_signals(sig, weights, k=2, min_coverage=0.8)

    assert rep["score"] == pytest.approx(-5)
    assert rep["coverage"] == pytest.approx(0.8)
    assert rep["label"] == "neutral"
    assert rep["drivers"] == {"positive": [("a", 25.0)], "negative": [("b", -30.0)]}

    strict = scoring.report_from_signals(sig, weights, k=2, min_coverage=0.9)
    assert strict["label"] == "insufficient data"


def test_render_score_uses_the_given_title_and_defaults_to_traditional():
    rep = {"score": 12.34, "label": "neutral", "coverage": 0.8,
           "drivers": {"positive": [("trend", 17.8)], "negative": [("volume", -3.2)]}}
    names = {"trend": "BTC trend", "volume": "Spot volume"}

    crypto_text = scoring.render_score(rep, names, title="Crypto regime")
    default_text = scoring.render_score(rep, names)

    assert crypto_text.split("\n") == [
        "Crypto regime: +12.3 (neutral)  coverage 80%",
        "Pushing up:   BTC trend +17.8",
        "Pushing down: Spot volume -3.2",
    ]
    assert default_text.split("\n")[0] == "Traditional regime: +12.3 (neutral)  coverage 80%"