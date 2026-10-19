import numpy as np
import pandas as pd

from momentum import report


def test_render_html_table_rows_labels_stale_and_dead():
    table = pd.DataFrame(
        {"last": [5.10, 7659.44, np.nan],
         "last_date": [pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-09"), pd.NaT],
         "age_days": [1, 7, np.nan],
         "chg_1": [-9.0, 0.0013, np.nan]},
        index=["DGS10", "SPX", "OLD"])
    units = {"DGS10": "bp", "SPX": "pct", "OLD": "pct"}
    labels = {"DGS10": "US 10y yield", "SPX": "S&P 500"}

    out = report.render_html_table(table, units, windows=(1,), stale_after=3, labels=labels)

    assert '<tr><td>US 10y yield</td><td>5.10</td><td>15 Oct</td><td>-9.0 bp</td></tr>' in out
    assert '<tr class="stale"><td>S&amp;P 500</td><td>7659.44</td><td>09 Oct</td><td>+0.13%</td></tr>' in out
    assert '<tr class="dead"><td>OLD</td><td colspan="3">unavailable</td></tr>' in out


def test_render_html_score_wraps_the_text_report_and_escapes():
    rep = {"score": -16.5, "label": "neutral", "coverage": 0.85,
           "drivers": {"positive": [("equities", 10.0)], "negative": []}}
    names = {"equities": "Stocks & bonds"}

    out = report.render_html_score(rep, names)

    assert out.startswith('<section class="regime" data-label="neutral">')
    assert "<p>Traditional regime: -16.5 (neutral)  coverage 85%</p>" in out
    assert "<p>Pushing up:   Stocks &amp; bonds +10.0</p>" in out
    assert "<p>Pushing down: none</p>" in out
    assert out.endswith("</section>")


def test_render_page_wraps_blocks_in_a_full_document():
    page = report.render_page(
        title="Morning Dashboard",
        generated="2026-10-18 08:00 UTC",
        blocks=["<p>A</p>", "<p>B</p>"],
        notes=["Source: FRED & Yahoo"],
    )

    assert page.startswith("<!doctype html>")
    assert "<title>Morning Dashboard</title>" in page
    assert "Generated 2026-10-18 08:00 UTC" in page
    assert page.index("<p>A</p>") < page.index("<p>B</p>")
    assert "<li>Source: FRED &amp; Yahoo</li>" in page
    assert 'name="viewport"' in page

def test_render_html_yield_wraps_the_two_lines_and_escapes():
    rep = {"regime": "growth scare",
           "changes": {"real_yield": -14.0, "breakeven": 0.0, "equities": -0.0195},
           "directions": {"real_yield": -1, "breakeven": 0, "equities": -1}}

    out = report.render_html_yield(rep)

    assert out.startswith('<section class="regime yield" data-label="growth scare">')
    assert "<p>Yield regime (draft rules, to be confirmed): growth scare</p>" in out
    assert "<p>10y real yield -14.0 bp (down), 10y breakeven +0.0 bp (flat), S&amp;P 500 -1.95% (down)</p>" in out
    assert out.endswith("</section>")