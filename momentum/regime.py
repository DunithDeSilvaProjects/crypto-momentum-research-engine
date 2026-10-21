import numpy as np
import pandas as pd
from momentum.snapshot import format_change, obs_change


def direction(change: float, dead: float) -> float:
    if pd.isna(change):
        return np.nan
    if abs(change) < dead:
        return 0
    return 1 if change > 0 else -1

def yield_regime(real: float, breakeven: float, equities: float) -> str:
    if any(pd.isna(x) for x in (real, breakeven, equities)):
        return "unavailable"
    if breakeven > 0 and equities < 0:
        return "inflationary risk-off"
    if real < 0 and breakeven <= 0 and equities < 0:
        return "growth scare"
    if real > 0 and breakeven >= 0 and equities > 0:
        return "growth-positive"
    if real < 0 and breakeven >= 0 and equities > 0:
        return "liquidity-positive"
    return "mixed"

YIELD_INPUTS = {
    "real_yield": ("DFII10", "bp"),
    "breakeven": ("T10YIE", "bp"),
    "equities": ("SP500", "pct"),
}


def yield_regime_report(df, n=5, dead_bp=2.0, dead_pct=0.005) -> dict:
    changes, dirs = {}, {}
    cols = [col for col, _ in YIELD_INPUTS.values() if col in df.columns]
    asof = df[cols].dropna().index.max()
    df = df[df.index <= asof]
    for name, (col, unit) in YIELD_INPUTS.items():
        change = obs_change(df[col], n, unit) if col in df.columns else np.nan
        dead = dead_bp if unit == "bp" else dead_pct
        changes[name] = change
        dirs[name] = direction(change, dead)
    return {
        "regime": yield_regime(dirs["real_yield"], dirs["breakeven"], dirs["equities"]),
        "changes": changes,
        "directions": dirs,
        "asof": asof,
    }

_NAMES = {"real_yield": "10y real yield", "breakeven": "10y breakeven", "equities": "S&P 500"}
_UNITS = {name: unit for name, (col, unit) in YIELD_INPUTS.items()}
_WORDS = {1: "up", 0: "flat", -1: "down"}


def render_yield_regime(rep: dict) -> str:
    parts = []
    for name, change in rep["changes"].items():
        word = _WORDS.get(rep["directions"][name], "unknown")
        parts.append(f"{_NAMES[name]} {format_change(change, _UNITS[name])} ({word})")
    head = f"Yield regime (draft rules, to be confirmed): {rep['regime']}"
    body = ", ".join(parts)
    asof = rep.get("asof")
    if asof is not None and not pd.isna(asof):
        body += f", as of {asof:%d %b}"
    return head + "\n" + body