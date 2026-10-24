import numpy as np
import pandas as pd
from momentum.snapshot import obs_change


def signal(change: float, normal: float, up_is_good: bool) -> float:
    if pd.isna(change) or pd.isna(normal) or normal == 0:
        return np.nan
    s = np.clip(change / normal, -2, 2) / 2
    return s if up_is_good else -s

def change_series(series: pd.Series, n: int, unit: str) -> pd.Series:
    s = series.dropna()
    if unit == "bp":
        return (s - s.shift(n)) * 100
    if unit == "pct":
        return s / s.shift(n) - 1
    raise ValueError(f"unit must be 'bp' or 'pct', got {unit!r}")

def normal_size(series: pd.Series, n: int, unit: str, lookback: int) -> float:
    ch = change_series(series, n, unit).dropna()
    past = ch.iloc[:-1].tail(lookback)
    if past.empty:
        return np.nan
    return past.abs().mean()

def component_signal(series: pd.Series, n: int, unit: str, up_is_good: bool, lookback: int) -> float:
    change = obs_change(series, n, unit)
    normal = normal_size(series, n, unit, lookback)
    return signal(change, normal, up_is_good)


def contributions(signals: dict, weights: dict) -> pd.Series:
    return pd.Series({
        k: weights[k] * (0 if pd.isna(signals[k]) else signals[k])
        for k in weights
    })

def regime_label(score: float) -> str:
    if pd.isna(score):
        return "unavailable"
    if score <= -60:
        return "strongly risk-off"
    if score <= -20:
        return "cautious"
    if score < 20:
        return "neutral"
    if score < 60:
        return "constructive"
    return "strongly risk-on"

def guarded_label(score: float, cov: float, min_coverage: float) -> str:
    if cov < min_coverage:
        return "insufficient data"
    return regime_label(score)

def component_signals(df: pd.DataFrame, spec: dict, n: int, lookback: int) -> dict:
    out = {}
    for name, c in spec.items():
        if c["col"] not in df.columns:
            out[name] = np.nan
            continue
        out[name] = component_signal(df[c["col"]], n, c["unit"], c["up_is_good"], lookback)
    return out

def coverage(signals: dict, weights: dict) -> float:
    total = sum(weights.values())
    covered = sum(w for name, w in weights.items() if not np.isnan(signals[name]))
    return covered / total

def top_drivers(contribs: pd.Series, k: int = 2) -> dict:
    pos = contribs[contribs > 0].sort_values(ascending=False).head(k)
    neg = contribs[contribs < 0].sort_values().head(k)
    return {"positive": list(pos.items()), "negative": list(neg.items())}

def report_from_signals(signals: dict, weights: dict, k: int = 2, min_coverage: float = 0.8) -> dict:
    contribs = contributions(signals, weights)
    score = contribs.sum()
    cov = coverage(signals, weights)
    return {
        "score": score,
        "coverage": cov,
        "label": guarded_label(score, cov, min_coverage),
        "drivers": top_drivers(contribs, k),
        "signals": signals,
    }

def traditional_report(df, spec, n, lookback, k=2, min_coverage=0.8) -> dict:
    sig = component_signals(df, spec, n, lookback)
    weights = {name: c["weight"] for name, c in spec.items()}
    return report_from_signals(sig, weights, k, min_coverage)

def stale_columns(df: pd.DataFrame, max_age_days: int) -> list:
    asof = df.index.max()
    stale = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty or np.busday_count(s.index[-1].date(), asof.date()) > max_age_days:
            stale.append(col)
    return stale

def render_score(report: dict, names: dict | None = None) -> str:
    names = names or {}

    def fmt(items):
        if not items:
            return "none"
        return ", ".join(f"{names.get(key, key)} {value:+.1f}" for key, value in items)

    head = (f"Traditional regime: {report['score']:+.1f} ({report['label']})"
            f"  coverage {report['coverage']:.0%}")
    return "\n".join([
        head,
        f"Pushing up:   {fmt(report['drivers']['positive'])}",
        f"Pushing down: {fmt(report['drivers']['negative'])}",
    ])