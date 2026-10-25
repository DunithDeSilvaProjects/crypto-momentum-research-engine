# Cross-sectional momentum in liquid crypto perpetuals: a null result

**Result: not supported.** A volatility-scaled trend signal (30/60/90-day) does not rank next-week returns of 20 liquid Binance perpetuals better than chance over 2021-2026. The pre-registered primary hypothesis fails criteria 1 and 3 of `research_protocol.md`.

## Question and protocol
Does trailing volatility-scaled price momentum rank the next days' returns of liquid crypto perpetual futures? The protocol (`docs/research_protocol.md`) was committed before any result was computed. It fixes the signal, the horizons, the execution lag, the metrics and the success criteria. Two amendments, both dated before any IC was computed, are listed in that file.

## Data and method
- 20 Binance USDT-M perpetuals, daily candles, 1 Jan 2021 to 24 Oct 2026 (2,123 days). ARB joins on 23 Mar 2023. No other gaps.
- Signal at the close of day t: log return over L days divided by (EWMA daily volatility x sqrt(L)), averaged over L = 30, 60, 90, winsorised at +/-3, then z-scored across liquid coins (30-day median quote volume of at least 10 million USD).
- The return starts one day later, at the close of t+1, and runs h days. The signal function was tested to use no future data.
- Metric: daily cross-sectional Spearman rank IC. The t-statistic uses non-overlapping samples (every h-th day), because overlapping h-day returns are not independent.
- 12 pre-registered hypotheses: 4 signals (combined, 30d, 60d, 90d) at 1, 5 and 10 days. Primary: combined, 5 days. Secondary hypotheses need |t| above 2.9 (Bonferroni).
- Placebo: a random signal on the same coins and days.

## Results
| signal | h | n | mean IC | t | hit rate | IC first half | IC second half |
|---|---|---|---|---|---|---|---|
| combined | 1 | 2031 | -0.0059 | -0.90 | 0.497 | -0.0113 | -0.0005 |
| **combined** | **5** | **406** | **-0.0020** | **-0.13** | **0.498** | **-0.0116** | **+0.0110** |
| combined | 10 | 203 | -0.0013 | -0.06 | 0.512 | -0.0013 | +0.0064 |
| 30d | 1 | 2091 | -0.0129 | -2.03 | 0.488 | -0.0096 | -0.0162 |
| 30d | 5 | 418 | -0.0064 | -0.45 | 0.522 | -0.0006 | -0.0085 |
| 30d | 10 | 209 | -0.0045 | -0.22 | 0.498 | +0.0077 | -0.0033 |
| 60d | 1 | 2061 | -0.0054 | -0.82 | 0.495 | -0.0129 | +0.0022 |
| 60d | 5 | 412 | -0.0074 | -0.50 | 0.468 | -0.0245 | +0.0111 |
| 60d | 10 | 206 | -0.0078 | -0.38 | 0.519 | -0.0181 | +0.0045 |
| 90d | 1 | 2031 | +0.0022 | +0.34 | 0.498 | -0.0046 | +0.0091 |
| 90d | 5 | 406 | +0.0041 | +0.27 | 0.527 | +0.0004 | +0.0120 |
| 90d | 10 | 203 | +0.0039 | +0.18 | 0.512 | +0.0138 | +0.0008 |
| placebo | 1 | 2031 | -0.0002 | -0.04 | 0.502 | +0.0005 | -0.0009 |
| placebo | 5 | 406 | +0.0114 | +0.92 | 0.522 | +0.0175 | +0.0279 |
| placebo | 10 | 203 | +0.0250 | +1.48 | 0.547 | +0.0072 | -0.0233 |

Reproduce with `PYTHONPATH=. python scripts/run_stage1.py`; the raw table is `docs/stage1_results.csv`.

## Reading
- Primary: mean IC -0.002, t = -0.13. Criterion 1 (t above 2) fails. The mean IC is negative in the first half and positive in the second, so criterion 3 (positive in both halves) also fails. Because the criteria must all hold, the hypothesis is not supported without needing criterion 2.
- No secondary hypothesis comes near |t| of 2.9. The largest, the 30-day signal at one day, is -2.03: the wrong sign for momentum, and about what one expects as the largest of 12 noisy tests.
- The placebo behaves like noise (all |t| below 1.5), so the harness does not create signal.
- Power: the standard error of the primary mean IC is about 0.015, giving a 95% interval of about -0.032 to +0.028. This rules out an IC above about 0.03 at five days. It cannot rule out the smaller ICs (0.02 to 0.03) that real signals often have. This is absence of evidence in this sample, not evidence that momentum does not exist.

## Limits
- Survivorship: the 20 names are today's survivors, chosen with hindsight.
- Small cross-section (about 20 coins) and one exchange's perpetuals.
- Rank IC measures ordering, not tradable profit. Stage 2 (long-short portfolio, costs, funding, baselines) was not run for this version. Criterion 2 was therefore not evaluated. It does not change the verdict, since criteria 1 and 3 already fail.
- One execution lag (one day) and one liquidity filter were examined. Neither was tuned.

## What would be a new study
Reversal, different lookbacks, other horizons, other universes or non-price signals are new hypotheses. They need their own protocol and fresh data, and must be labelled exploratory if looked at on this sample.

## Addendum, 25 Oct 2026: Stage 2 (portfolio, costs, funding, baselines)

Run exactly as fixed in the protocol amendment of the same date: combined signal, 5-day horizon, long the top 5 and short the bottom 5, 10 bp per side, actual Binance funding, evaluation window 2021-04-03 to 2026-10-24 (2,031 days, about 5.6 years). Nothing was changed after the numbers were seen.

| Series | Gross Sharpe | Net of costs | Net of costs and funding | Std. error | Turnover per year |
|---|---|---|---|---|---|
| Long-short momentum | 0.08 | -0.04 | -0.01 | 0.42 | 30.4 |
| Equal-weight long-only | 0.37 | 0.37 | 0.29 | 0.43 | 0.2 |
| BTC buy-and-hold | 0.38 | 0.38 | 0.23 | 0.43 | 0.0 |

Reading
- Success criterion 2 is not met: the long-short Sharpe after costs and funding is -0.01 (standard error 0.42), which is not positive and is indistinguishable from zero. Criteria 1 and 3 had already failed in Stage 1, so all three criteria fail and the null result stands.
- The gross Sharpe of 0.08 is what an IC of about zero implies. The book turns over about 30 times its gross exposure a year, roughly 3.0% of capital in costs at 10 bp per side, which takes the Sharpe from 0.08 to -0.04. Funding then adds back 0.03 for this dollar-neutral book.
- The long-only baselines earn Sharpe ratios of 0.23 to 0.29, each within about 0.7 standard errors of zero, and they carry crypto beta and survivorship. They are a reference point, not evidence for or against the signal. Funding costs them 0.08 (equal-weight) and 0.15 (BTC) of Sharpe.

Limits specific to Stage 2
- One path of 5.6 years: with a Sharpe standard error of about 0.4, only an annual Sharpe above roughly 0.8 could have been told apart from zero.
- Funding gaps are zero-filled by the pre-registered rule. Coverage is 100% for 18 of the 20 coins, 68% for BNB (probably a funding schedule change that drops the minority-schedule days; not investigated) and 62% for ARB (listed in 2023). BNB's funding is therefore understated.
- Costs are a flat 10 bp per side. There is no market impact or spread widening in stress, which would only make the net result worse.
- The 20 coins are today's survivors, which flatters the long-only baselines most.
