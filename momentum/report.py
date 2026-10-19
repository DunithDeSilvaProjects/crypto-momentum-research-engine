import html

import pandas as pd

from momentum.snapshot import format_change
from momentum import scoring,regime

CSS = """
:root{--bg:#fff;--fg:#1a1d23;--muted:#6b7280;--line:#e5e7eb;--neg:#b42318;--pos:#067647;--neu:#475467;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8ec;--muted:#9aa3b2;--line:#262b34;--neg:#f97066;--pos:#47cd89;--neu:#98a2b3;--warn:#fdb022}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0}
.generated{color:var(--muted);margin:4px 0 20px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
tr.stale td{color:var(--warn)}
tr.dead td{color:var(--muted)}
.regime{border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0}
.regime p{margin:4px 0}
.regime[data-label*="risk-off"],.regime[data-label="cautious"]{border-left:4px solid var(--neg)}
.regime[data-label*="risk-on"],.regime[data-label="constructive"]{border-left:4px solid var(--pos)}
.regime[data-label="neutral"]{border-left:4px solid var(--neu)}
.notes{color:var(--muted);font-size:13px}
"""

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

def render_html_score(rep: dict, names: dict | None = None) -> str:
    lines = scoring.render_score(rep, names).split("\n")
    body = "".join(f"<p>{html.escape(line)}</p>" for line in lines)
    label = html.escape(rep["label"], quote=True)
    return f'<section class="regime" data-label="{label}">{body}</section>'

def render_page(title: str, generated: str, blocks: list, notes: list) -> str:
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f'<h1>{html.escape(title)}</h1><p class="generated">Generated {html.escape(generated)}</p>'
        + "".join(f"<div>{block}</div>" for block in blocks)
        + f'<ul class="notes">{notes_html}</ul></body></html>'
    )

def render_html_yield(rep: dict) -> str:
    lines = regime.render_yield_regime(rep).split("\n")
    body = "".join(f"<p>{html.escape(line)}</p>" for line in lines)
    label = html.escape(rep["regime"], quote=True)
    return f'<section class="regime yield" data-label="{label}">{body}</section>'

def render_html_asof(text: str) -> str:
    body = "".join(f"<p>{html.escape(line)}</p>" for line in text.split("\n"))
    return f'<section class="asof">{body}</section>'