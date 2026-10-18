import html

import pandas as pd

from momentum.snapshot import format_change


def render_html_table(table, units, windows=(1, 5, 20), stale_after=3, labels=None) -> str:
    labels = labels or {}
    heads = "".join(f"<th>{n}d</th>" for n in windows)
    rows = []
    for name, r in table.iterrows():
        label = html.escape(labels.get(name, name))
        if pd.isna(r["last_date"]):
            span = len(windows) + 2
            rows.append(f'<tr class="dead"><td>{label}</td><td colspan="{span}">unavailable</td></tr>')
            continue
        cells = [f"<td>{label}</td>", f"<td>{r['last']:.2f}</td>", f"<td>{r['last_date']:%d %b}</td>"]
        for n in windows:
            change = format_change(r[f"chg_{n}"], units[name])
            cells.append(f"<td>{change}</td>")
        cls = ' class="stale"' if r["age_days"] > stale_after else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Series</th><th>Last</th><th>As of</th>" + heads
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")