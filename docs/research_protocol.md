# Research protocol: crypto cross-sectional momentum

Written 25 Oct 2026, before any rank IC, portfolio return or Sharpe ratio was computed for this study. Amendments after results are appended below with the date and the reason. They never replace this text.

## Question
Does trailing volatility-scaled price momentum rank the next days' returns of liquid crypto perpetual futures, net of trading costs and funding?

## Universe and sample
- 20 Binance USDT-M perpetuals (momentum.config.DEFAULT_SYMBOLS), daily candles from 2021-01-01 to the last complete day at run time.
- A coin enters the cross-section only after 91 days of history and only on days when its 30-day average quote volume is at least 10 million USD (liquidity mask).
- Known limitation: the 20 names are today's survivors, chosen with hindsight. Delisted coins are absent, which flatters any long-biased result. The note will say so and report how many coins are in the cross-section each year.

## Signal (fixed, nothing is fitted)
- Trend score: log return over L days divided by (EWMA daily volatility, half-life 30, times sqrt(L)), averaged over L = 30, 60, 90, winsorised at +/-3, then z-scored across coins each day.
- Variants counted for multiple testing: the combined signal and the three single lookbacks (4 signals), at 3 horizons (1, 5, 10 days) = 12 hypotheses.

## Timing
- The signal is formed with data up to the close of day t. The trade happens at the close of day t+1 (a one-day execution lag) and is held for h days. The forward return is close(t+1+h) / close(t+1) - 1.

## Primary hypothesis (one, stated in advance)
The combined signal at the 5-day horizon has a positive mean daily cross-sectional rank IC.

## Metrics
- Mean Spearman rank IC and its t-statistic from non-overlapping samples (every h-th day), hit rate.
- Long-short portfolio: long the top 5 and short the bottom 5 coins by signal, equal weights, dollar neutral, gross exposure 1, rebalanced every h days.
- Costs: 10 bp per side on traded notional (fee plus slippage). Funding: actual Binance funding paid and received on each position.
- Baselines: equal-weight long-only portfolio of the same coins, and Bitcoin buy-and-hold.

## Success criteria (decided now)
The primary hypothesis is supported only if ALL hold:
1. Mean rank IC is above zero with t above 2.
2. The long-short portfolio has a positive Sharpe ratio after costs and funding.
3. The mean IC is positive in both halves of the sample.
Secondary hypotheses (the other 11) must clear a Bonferroni-adjusted bar of t above 2.9.
If the criteria are not met, the note reports a null result and says so in its first line.

## Not allowed
No tuning of lookbacks, thresholds, costs or horizons on this sample. No dropping of periods after seeing them.

## Not tested here
Whether the dashboard's regime scores predict anything. They are descriptive, rules-based summaries and are not claimed to be forecasts.

## Amendments
- 25 Oct 2026, before any rank IC was computed: the liquidity filter uses the 30-day MEDIAN of daily quote volume (at least 10 million USD), which is what the tested engine function liquidity_mask computes, instead of the average written above. A median is not moved by one spike day. The 91-day history rule needs no separate code: the signal needs 90 days of prices, so a coin has no signal before that.
- 25 Oct 2026, before any rank IC was computed: the evaluation is staged. Stage 1 (rank IC, t-statistics, half-sample check) is run first. Stage 2 (long-short portfolio, costs, funding, baselines) follows. Success criterion 2 is evaluated only in stage 2. Until then the primary hypothesis is reported as IC evidence only, not yet assessed against all three criteria.
- 25 Oct 2026, before any portfolio return, Sharpe ratio, turnover or funding figure was computed: Stage 2 is fixed as follows.
  - Scope: the primary hypothesis only (combined signal, 5-day horizon). No other signal or horizon is run through the portfolio, so Stage 2 adds no hypotheses.
  - Portfolio: long the 5 highest and short the 5 lowest signals, 0.1 per coin per leg, gross exposure 1, dollar neutral. A day with fewer than 10 coins carrying a signal has no position. Weights are taken from every 5th day of the panel and carried forward for 5 days, with the one-day execution lag.
  - Costs: 10 bp per side on traded notional (gross turnover times 0.001), charged on the day the new position starts earning.
  - Funding: actual Binance USDT-M funding history from 2021-01-01, summed per UTC day for each coin. Longs pay and shorts receive when the rate is positive. Days with an incomplete set of fundings for a coin (a partial day or a schedule change) and days before a coin's listing carry zero funding. This understates funding on those days and is reported as a limitation.
  - Baselines, with the same lag, cost and funding treatment: (a) equal-weight long-only over all coins with a signal, gross 1, rebalanced every 5 days; (b) BTCUSDT buy-and-hold, weight 1 from the first day of the evaluation window.
  - Window: from the first day on which the long-short portfolio holds a position to the last complete day, the same for all three series.
  - Reported: annualised Sharpe ratio (365 days, sample standard deviation) of the strategy gross, net of costs, and net of costs and funding (the series used for criterion 2); net Sharpe of both baselines; mean daily turnover and its annual equivalent; the approximate standard error of a Sharpe ratio, sqrt((1 + SR^2 / 2) / years).
  - Status: informational. Stage 1 already failed criteria 1 and 3 and the criteria are conjunctive, so nothing in Stage 2 can change the null verdict on the primary hypothesis. No parameter is changed after the Stage 2 numbers are seen.
