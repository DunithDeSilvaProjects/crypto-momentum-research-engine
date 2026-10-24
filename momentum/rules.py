from momentum.config import DEFAULT_SYMBOLS, SignalConfig
from momentum.crypto import CRYPTO_MIN_COVERAGE, CRYPTO_WEIGHTS
STATUSES = ("convention", "judgment", "proxy")

RULES = [
    {"area": "Score components",
     "rule": "Six components (equities, VIX, HY credit spreads, dollar, 10y real yield, 2s10s curve), each the 5-observation change compared with its normal size over the last 60 observations.",
     "status": "judgment"},
    {"area": "Window",
     "rule": "The score and the yield regime use the 5-observation change (about one trading week). 1-day and 20-day changes are shown in the table only.",
     "status": "judgment"},
    {"area": "Weights",
     "rule": "Equities 20, real yield 20, VIX 15, HY spreads 15, dollar 15, curve 15. Not validated.",
     "status": "judgment"},
    {"area": "Curve",
     "rule": "Curve is the 10y minus 2y yield (T10Y2Y); steepening counts as risk-on. The sign of short-term curve changes is ambiguous (bull versus bear steepening).",
     "status": "convention"},
    {"area": "Label bands",
     "rule": "Score <= -60 strongly risk-off, <= -20 cautious, below 20 neutral, below 60 constructive, otherwise strongly risk-on.",
     "status": "judgment"},
    {"area": "Minimum coverage",
     "rule": "If less than 80% of the weight has data, no regime label is shown.",
     "status": "judgment"},
    {"area": "Staleness",
     "rule": "A series more than 3 business days older than the newest date is excluded from the score and flagged in the table. Exchange holidays are not modelled.",
     "status": "judgment"},
    {"area": "Market cut-off",
     "rule": "Only fully closed US sessions are used: after 17:00 New York time the day counts, and weekends roll back to Friday.",
     "status": "judgment"},
    {"area": "Yield regime",
     "rule": "Classifies the moves of the 10y real yield, 10y breakeven and S&P 500 (nominal yield = real yield + breakeven inflation) into inflationary risk-off, growth scare, growth-positive, liquidity-positive, or mixed.",
     "status": "convention"},
    {"area": "Yield-regime thresholds",
     "rule": "A move is flat inside 2 bp (yields) or 0.5% (S&P 500), computed on the latest date all three inputs share. Draft values, to be replaced by data-driven thresholds.",
     "status": "judgment"},
    {"area": "Equity breadth",
     "rule": "Not included yet. Planned free proxy: equal-weight versus cap-weight S&P 500.",
     "status": "proxy"},
]


def crypto_rules() -> list:
    cfg = SignalConfig()
    weights = ", ".join(f"{name.replace('_', ' ')} {weight}" for name, weight in CRYPTO_WEIGHTS.items())
    lookbacks = ", ".join(str(n) for n in cfg.lookbacks)
    return [
        {"area": "Crypto components",
         "rule": (f"Eight components, weights from the brief: {weights}. ETF flows and liquidations have "
                  "no free source, so they are shown as not scored and coverage cannot exceed 80%."),
         "status": "convention"},
        {"area": "Crypto minimum coverage",
         "rule": (f"No label below {CRYPTO_MIN_COVERAGE:.0%} coverage. Lower than the traditional 80% because "
                  "20% of the weight is permanently unavailable: one further failed 10-point component still "
                  "gives a label, two do not."),
         "status": "judgment"},
        {"area": "Crypto windows",
         "rule": (f"Calendar days (crypto trades every day), not trading days. Trend: average of the {lookbacks}-day "
                  f"volatility-scaled momentum of Bitcoin, capped at +/-{cfg.winsor:g} and scaled to +/-1. "
                  f"Breadth: share of the {len(DEFAULT_SYMBOLS)} coins with positive trend, scaled to +/-1. "
                  "Volume: latest day against the previous 20, signed by the day's price direction, floored at zero."),
         "status": "judgment"},
        {"area": "Crypto derivatives",
         "rule": ("Funding: contrarian, average of the last 3 complete days against a base of 0.03% a day; "
                  "0.06% above it is full strength. Open interest: 5-day change (+10% is full strength) "
                  "signed by the 5-day price direction; falling open interest scores zero. "
                  "Stablecoin supply: 7-day change, +/-1% is full strength."),
         "status": "judgment"},
        {"area": "Crypto data",
         "rule": ("Prices and volume: Binance spot daily candles (UTC days), unfinished day dropped. Funding and "
                  "open interest: OKX BTC-USDT perpetual swap; its daily open-interest bars run on a UTC+8 day, "
                  "so they are 8 hours offset from the spot candles. Stablecoin supply: DefiLlama, complete UTC "
                  "days only. The research engine uses Binance perpetual candles; the dashboard uses spot "
                  "because Binance futures block GitHub's servers."),
         "status": "proxy"},
    ]


RULES = RULES + crypto_rules()
