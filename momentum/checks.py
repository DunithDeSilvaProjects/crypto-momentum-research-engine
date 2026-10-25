import numpy as np
import pandas as pd


def identity_check(df: pd.DataFrame, a: str, b: str, c: str, tol: float = 0.02) -> dict:
    unknown = {"asof": pd.NaT, "gap": np.nan, "ok": None}
    if not all(x in df.columns for x in (a, b, c)):
        return unknown
    rows = df[[a, b, c]].dropna()
    if rows.empty:
        return unknown
    last = rows.iloc[-1]
    gap = last[a] - last[b] - last[c]
    return {"asof": rows.index[-1], "gap": gap, "ok": bool(abs(gap) <= tol)}


def render_checks(checks: list) -> str:
    lines = []
    for label, result in checks:
        if result["ok"] is None:
            lines.append(f"{label}: not checked (data missing)")
            continue
        verdict = "ok" if result["ok"] else "MISMATCH"
        lines.append(f"{label}: {verdict} (gap {result['gap'] * 100:+.1f} bp, as of {result['asof']:%d %b})")
    return "\n".join(lines)