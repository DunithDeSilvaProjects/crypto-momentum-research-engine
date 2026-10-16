import numpy as np
import pandas as pd


def obs_change(series: pd.Series, n: int, unit: str) -> float:
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    last, past = s.iloc[-1], s.iloc[-n-1]
    if unit == "bp":
        return (last - past) * 100
    if unit == "pct":
        return last / past - 1
    raise ValueError(f"unit must be 'bp' or 'pct', got {unit!r}")

def snapshot_table(df: pd.DataFrame, units: dict, windows=(1, 5, 20)) -> pd.DataFrame:
    asof = df.index.max()
    rows = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            rows[col] = {
                "last": np.nan,
                "last_date": pd.NaT,
                "age_days": np.nan,
                **{f"chg_{n}": np.nan for n in windows},
            }
            continue
        row = {
            "last": s.iloc[-1],
            "last_date": s.index[-1],
            "age_days": (asof - s.index[-1]).days,
        }
        for n in windows:
            row[f"chg_{n}"] = obs_change(s, n, units[col])
        rows[col] = row
    return pd.DataFrame.from_dict(rows, orient="index")

def format_change(x: float, unit: str) -> str:
    if pd.isna(x):
        return "n/a"
    if unit == "bp":
        return f"{x:+.1f} bp"
    if unit == "pct":
        return f"{x * 100:+.2f}%"
    raise ValueError(f"unit must be 'bp' or 'pct', got {unit!r}")

def render_text(table: pd.DataFrame, units: dict, windows=(1, 5, 20), stale_after: int = 3) -> str:
    out = []                                              # will hold one text line per series
    for name, r in table.iterrows():                      # name = "US10", r = that row's values
        parts = [f"{name:<8}{r['last']:>10.2f} ({r['last_date']:%d %b})"]
        for n in windows:                                 # one piece per window: "1d -3.0 bp"
            parts.append(f"{n}d {format_change(r[f'chg_{n}'], units[name])}")
        line = "  ".join(parts)                           # glue the pieces with two spaces
        if r["age_days"] > stale_after:                    # too old? add the warning
            line += f"  STALE ({int(r['age_days'])}d old)"
        out.append(line)
    return "\n".join(out)                                 # all lines into one text block
