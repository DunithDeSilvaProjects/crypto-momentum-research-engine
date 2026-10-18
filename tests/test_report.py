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