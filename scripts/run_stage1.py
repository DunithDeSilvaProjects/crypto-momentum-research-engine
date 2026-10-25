"""Stage 1 of docs/research_protocol.md: rank IC of the momentum signal.

Run from the repo root:  PYTHONPATH=. python scripts/run_stage1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from momentum import research
from momentum.config import SignalConfig
from momentum.data import build_default_panel

HORIZONS = (1, 5, 10)
VARIANTS = {"combined": (30, 60, 90), "30d": (30,), "60d": (60,), "90d": (90,)}


def evaluate(name, sig, close, horizon):
    fwd = research.forward_returns(close, horizon=horizon, lag=1)
    ic = research.rank_ic(sig, fwd).dropna()
    whole = research.ic_summary(ic, step=horizon)
    middle = ic.index[len(ic) // 2]
    first = research.ic_summary(ic[ic.index < middle], step=horizon)
    second = research.ic_summary(ic[ic.index >= middle], step=horizon)
    return {"signal": name, "h": horizon, "n": whole["n"], "mean_ic": whole["mean"],
            "t": whole["t"], "hit": whole["hit_rate"],
            "first_half": first["mean"], "second_half": second["mean"]}


def main():
    panel = build_default_panel("2021-01-01")
    rows = []
    for name, lookbacks in VARIANTS.items():
        sig = research.momentum_signal(panel.close, panel.quote_volume, SignalConfig(lookbacks=lookbacks))
        for horizon in HORIZONS:
            rows.append(evaluate(name, sig, panel.close, horizon))

    combined = research.momentum_signal(panel.close, panel.quote_volume)
    rng = np.random.default_rng(42)
    noise = pd.DataFrame(rng.normal(size=combined.shape), index=combined.index, columns=combined.columns)
    noise = noise.where(combined.notna())
    for horizon in HORIZONS:
        rows.append(evaluate("placebo", noise, panel.close, horizon))

    table = pd.DataFrame(rows)
    print("Sample:", panel.close.index[0].date(), "to", panel.close.index[-1].date())
    print(table.round(4).to_string(index=False))
    print()
    print("PRIMARY hypothesis (combined signal, 5-day horizon):")
    print(table.query("signal == 'combined' and h == 5").round(4).to_string(index=False))
    print("Secondary hypotheses must clear |t| > 2.9 (Bonferroni, 12 hypotheses).")
    Path("docs").mkdir(exist_ok=True)
    table.to_csv("docs/stage1_results.csv", index=False)
    print("Wrote docs/stage1_results.csv")


if __name__ == "__main__":
    main()
