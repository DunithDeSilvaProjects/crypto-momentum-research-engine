"""Stage 2 of docs/research_protocol.md: long-short portfolio, costs, funding, baselines.

Run from the repo root:  PYTHONPATH=. python scripts/run_stage2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from momentum import crypto, research
from momentum.data import build_default_panel

K = 5
HORIZON = 5
LAG = 1
COST_BPS = 10.0
PERIODS = 365
FUNDING_START_MS = int(pd.Timestamp("2021-01-01").timestamp() * 1000)


def summarise(name, frame, start):
    window = frame.loc[start:]
    years = len(window) / 365
    net_sharpe = research.sharpe(window["net"], PERIODS)
    return {
        "series": name,
        "days": len(window),
        "gross_sharpe": research.sharpe(window["gross"], PERIODS),
        "net_of_cost_sharpe": research.sharpe(window["net"] - window["funding"], PERIODS),
        "net_sharpe": net_sharpe,
        "sharpe_se": float(np.sqrt((1 + net_sharpe ** 2 / 2) / years)),
        "turnover_per_day": window["turnover"].mean(),
        "turnover_per_year": window["turnover"].mean() * PERIODS,
    }


def main():
    panel = build_default_panel("2021-01-01")
    close = panel.close
    signal = research.momentum_signal(close, panel.quote_volume)

    funding = crypto.funding_panel(list(close.columns), FUNDING_START_MS, close.index)
    print("Share of days with non-zero funding, by coin:")
    print((funding != 0).mean().round(2).to_string())
    assert funding.abs().to_numpy().sum() > 0, "funding table is all zero: index mismatch?"

    btc_weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    btc_weights["BTCUSDT"] = 1.0
    books = {
        "long-short momentum": research.long_short_weights(signal, K),
        "equal-weight long-only": research.equal_weight_weights(signal),
        "BTC buy-and-hold": btc_weights,
    }
    runs = {
        name: research.run_portfolio(w, close, funding, HORIZON, LAG, COST_BPS)
        for name, w in books.items()
    }

    start = runs["long-short momentum"]["turnover"].ne(0).idxmax()
    table = pd.DataFrame([summarise(name, run, start) for name, run in runs.items()])

    print()
    print("Window:", start.date(), "to", close.index[-1].date())
    print(table.round(4).to_string(index=False))
    print()
    print("Informational: Stage 1 failed criteria 1 and 3, so this cannot change the null verdict.")
    Path("docs").mkdir(exist_ok=True)
    table.to_csv("docs/stage2_results.csv", index=False)
    print("Wrote docs/stage2_results.csv")


if __name__ == "__main__":
    main()