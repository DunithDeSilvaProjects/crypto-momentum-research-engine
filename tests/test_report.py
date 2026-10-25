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

def test_render_html_asof_makes_one_paragraph_per_line_in_an_asof_section():
    text = "Generated: 20 Oct 00:34 UTC\nMarket data through: Mon 19 Oct 2026 & more"
    out = report.render_html_asof(text)

    assert out.startswith('<section class="asof">')
    assert out.count("<p>") == 2
    assert "<p>Generated: 20 Oct 00:34 UTC</p>" in out
    assert "Mon 19 Oct 2026 &amp; more" in out                       # the ampersand must be escaped


def test_render_html_rules_makes_one_row_per_rule_with_its_status():
    rules = [
        {"area": "Window", "rule": "5-observation change", "status": "judgment"},
        {"area": "Yield regime", "rule": "real & breakeven vs equities", "status": "convention"},
    ]
    out = report.render_html_rules(rules)

    assert out.startswith('<details class="rules">')
    assert out.count("<tr>") == 3                 # header row plus one per rule
    assert 'data-status="judgment"' in out
    assert "real &amp; breakeven" in out


def test_render_html_score_passes_the_title_through():
    rep = {"score": 12.34, "label": "neutral", "coverage": 0.8,
           "drivers": {"positive": [], "negative": []}}

    page = report.render_html_score(rep, title="Crypto regime")

    assert "Crypto regime: +12.3 (neutral)" in page
    assert 'data-label="neutral"' in page


def test_render_html_checks_wraps_each_line_in_a_paragraph_and_escapes():
    out = report.render_html_checks("A & B: ok\nC: MISMATCH")

    assert out == '<section class="checks"><p>A &amp; B: ok</p><p>C: MISMATCH</p></section>'