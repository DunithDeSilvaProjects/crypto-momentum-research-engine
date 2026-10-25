# Crypto Momentum Research Engine

**Result: null.** Volatility-scaled trailing momentum did not rank the next days' returns of 20 liquid Binance USDT-M perpetuals over 2021-2026, and the long-short portfolio did not earn a positive Sharpe ratio after costs and funding. The protocol was committed before any result was computed and nothing was tuned afterwards.

| | Estimate | Uncertainty |
|---|---|---|
| Mean daily rank IC (combined 30/60/90-day signal, 5-day horizon, 1-day execution lag) | -0.0020 | t = -0.13; 95% interval about -0.032 to +0.028 |
| Long-short Sharpe (top 5 against bottom 5), gross | 0.08 | standard error about 0.42 |
| Same, after 10 bp per side and actual funding | -0.01 | standard error about 0.42 |

This is absence of evidence in one 5.6-year sample, not proof that momentum does not exist: the interval rules out an IC above about 0.03, not smaller effects.

- Write-up: [docs/research_note.md](docs/research_note.md)
- Pre-registered protocol and dated amendments: [docs/research_protocol.md](docs/research_protocol.md)
- Result tables: `docs/stage1_results.csv`, `docs/stage2_results.csv`

## Question

Does trailing volatility-scaled price momentum rank the next days' returns of liquid crypto perpetual futures, net of trading costs and funding? Order flow was not tested.

## Method

- **Universe:** 20 Binance USDT-M perpetuals, daily candles from 2021-01-01. A coin enters the cross-section only with 90 days of history and a 30-day median quote volume of at least 10 million USD.
- **Signal:** log return over 30, 60 and 90 days divided by EWMA volatility, averaged, winsorised, z-scored across coins each day. Nothing is fitted. The tests check that it has no look-ahead.
- **Stage 1 (rank IC):** Spearman rank IC, t-statistic from non-overlapping samples, hit rate, half-sample stability, a placebo random signal, and 12 hypotheses (4 signals, 3 horizons) with a Bonferroni bar of |t| > 2.9. None cleared it.
- **Stage 2 (portfolio):** long the top 5 and short the bottom 5, equal weight, dollar neutral, rebalanced every 5 days, one-day execution lag, 10 bp per side, actual Binance funding. Baselines: equal-weight long-only and BTC buy-and-hold.
- **Integrity:** the protocol (hypotheses, success criteria, what is not allowed) was committed before any IC or Sharpe was computed. Amendments are dated and appended, never edited in.

## Limits

- The 20 coins are today's survivors, which flatters long-only results.
- Costs are a flat 10 bp per side, with no market impact.
- Funding gaps are set to zero by a pre-registered rule, which understates funding for BNB in particular.
- One sample, one path: a Sharpe standard error of about 0.4 means only large effects could have been detected.

## Reproduce

~~~bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
pytest -q
PYTHONPATH=. python scripts/run_stage1.py
PYTHONPATH=. python scripts/run_stage2.py
~~~

Both research scripts use public Binance endpoints and need no API key. Stage 2 makes about 250 funding requests.

## Morning market dashboard

A separate part of the repository builds a daily descriptive snapshot: rules-based regime scores for traditional markets and for crypto, a yield-regime classification, a data table and a Rules box listing every assumption. It is not a forecast and the scores are not validated. A GitHub Actions workflow publishes it to GitHub Pages behind a password. Run it locally with `PYTHONPATH=. python scripts/morning_snapshot.py`. It needs a free FRED API key, which the workflow passes in as a secret.

## Layout

- `momentum/` library code: signals, research functions, dashboard scoring and rendering
- `scripts/` runnable entry points
- `tests/` the test suite
- `docs/` protocol, research note and result tables
- `.github/workflows/` tests and the scheduled dashboard
