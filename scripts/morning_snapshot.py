from momentum import snapshot
from momentum.macro import load_fred
from momentum import scoring
import pandas as pd
from momentum.market import drop_incomplete, fetch_yf_optional, last_complete_session, render_asof
from pathlib import Path
from momentum.config import DEFAULT_SYMBOLS
from momentum import report as pagebuilder
from momentum import brief, checks, crypto, regime, rules



UNITS = {
    "DGS10": "bp", "DGS2": "bp", "T10Y2Y": "bp", "DFII10": "bp", "T10YIE": "bp",
    "BAMLH0A0HYM2": "bp",
    "VIX": "pct", "SP500": "pct", "NASDAQCOM": "pct",
    "DCOILWTICO": "pct", "DXY": "pct", "GOLD": "pct", "NDX": "pct",
    "BTC": "pct", "ETH": "pct",
    "SX5E": "pct", "N225": "pct",
}
WINDOWS = (1, 5, 20)

LABELS = {
    "DGS10": "US 10y yield", "DGS2": "US 2y yield", "T10Y2Y": "2s10s curve",
    "DFII10": "10y real yield", "T10YIE": "10y breakeven", "BAMLH0A0HYM2": "HY credit spread",
    "VIX": "VIX", "SP500": "S&P 500", "NASDAQCOM": "Nasdaq Composite",
    "DCOILWTICO": "WTI crude", "DXY": "US Dollar Index (DXY)", "GOLD": "Gold futures", "NDX": "Nasdaq 100",
    "BTC": "Bitcoin (spot)", "ETH": "Ethereum (spot)",
    "SX5E": "Euro Stoxx 50", "N225": "Nikkei 225",
}
TRADITIONAL_SPEC = {
    "equities":   {"col": "SP500",        "unit": "pct", "up_is_good": True,  "weight": 20},
    "vix":        {"col": "VIX",          "unit": "pct", "up_is_good": False, "weight": 15},
    "spreads":    {"col": "BAMLH0A0HYM2", "unit": "bp",  "up_is_good": False, "weight": 15},
    "dollar":     {"col": "DXY",          "unit": "pct", "up_is_good": False, "weight": 15},
    "real_yield": {"col": "DFII10",       "unit": "bp",  "up_is_good": False, "weight": 20},
    "curve":      {"col": "T10Y2Y",       "unit": "bp",  "up_is_good": True,  "weight": 15},
}
SCORE_NAMES = {
    "equities": "Equity trend", "vix": "VIX", "spreads": "HY credit spreads",
    "dollar": "Dollar (DXY)", "real_yield": "10y real yield", "curve": "10y-2y curve",
}

CRYPTO_NAMES = {
    "trend": "BTC trend", "breadth": "Market breadth", "volume": "Spot volume",
    "funding": "Funding (contrarian)", "open_interest": "Open interest",
    "stablecoins": "Stablecoin supply",
    "etf_flows": "ETF flows (no free source)", "liquidations": "Liquidations (no free source)",
}

MAX_AGE_DAYS = 3
YF_SYMBOLS = {"DXY": "DX-Y.NYB", "GOLD": "GC=F", "NDX": "^NDX","VIX": "^VIX", "SX5E": "^STOXX50E", "N225": "^N225"}
COIN_NAMES = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
SITE_DIR = Path("site")

assert sum(c["weight"] for c in TRADITIONAL_SPEC.values()) == 100

def main() -> None:
    fred = load_fred([k for k in UNITS if k not in YF_SYMBOLS and k not in COIN_NAMES.values()], start="2025-01-01")
    now = pd.Timestamp.now(tz="UTC")
    cutoff = last_complete_session(now)
    yf_raw, yf_note = fetch_yf_optional(YF_SYMBOLS, "2025-01-01")
    yf_closes = drop_incomplete(yf_raw, cutoff)
    spot_closes, spot_volumes, spot_failed = crypto.load_spot_panel(DEFAULT_SYMBOLS, now)
    coin_rows = crypto.select_for_table(spot_closes, COIN_NAMES)
    df = pd.concat([fred, yf_closes, coin_rows], axis=1, sort=True)
    asof = render_asof(now, cutoff)
    print(asof)
    if yf_note:
        print(yf_note)
    if spot_failed:
        print("Binance spot data unavailable for: " + ", ".join(spot_failed))
    table = snapshot.snapshot_table(df, UNITS, windows=WINDOWS)
    print(snapshot.render_text(table, UNITS, windows=WINDOWS, stale_after=3, labels=LABELS))
    stale = scoring.stale_columns(df, MAX_AGE_DAYS)
    report = scoring.traditional_report(df.drop(columns=stale), TRADITIONAL_SPEC, n=5, lookback=60)
    print()
    print(scoring.render_score(report, SCORE_NAMES))
    derivs, deriv_notes = crypto.load_derivatives(now)
    crypto_rep, crypto_note = crypto.attempt(
        "Crypto score", lambda: crypto.crypto_report(spot_closes, spot_volumes, derivs)
    )
    if crypto_rep is not None:
        print()
        print(scoring.render_score(crypto_rep, CRYPTO_NAMES, "Crypto regime"))
    for note in deriv_notes + ([crypto_note] if crypto_note else []):
        print(note)
    if stale:
        print("Excluded as stale: " + ", ".join(LABELS.get(c, c) for c in stale))
    print("Source: FRED (daily). DXY, VIX, gold and Nasdaq 100 from Yahoo Finance via yfinance (unofficial, delayed).")
    generated = now.strftime("%Y-%m-%d %H:%M UTC")
    
    yrep = regime.yield_regime_report(df.drop(columns=stale))
    print()
    print(regime.render_yield_regime(yrep))
    check_text = "Data checks:\n" + checks.render_checks([
        ("10y minus 2y = 2s10s curve", checks.identity_check(df, "DGS10", "DGS2", "T10Y2Y")),
        ("10y minus real yield = breakeven", checks.identity_check(df, "DGS10", "DFII10", "T10YIE")),
    ])
    print()
    print(check_text)
    
    notes = ["Source: FRED (daily). DXY, VIX, gold and Nasdaq 100 from Yahoo Finance via yfinance (unofficial, delayed)."]
    if yf_note:
        notes.append(yf_note)
    if stale:
        notes.append("Excluded as stale: " + ", ".join(LABELS.get(c, c) for c in stale))
    notes.append("Crypto: Binance spot (data-api.binance.vision), OKX perpetual swaps (funding, open interest), DefiLlama (stablecoin supply).")
    notes.extend(deriv_notes)
    if crypto_note:
        notes.append(crypto_note)
    notes.append("Rules-based, descriptive score. Weights are not validated and it is not a forecast.")
    headline = brief.render_headline(report, crypto_rep, yrep, {**SCORE_NAMES, **CRYPTO_NAMES})
    print()
    print(headline)
    crypto_blocks = (
        [pagebuilder.render_html_score(crypto_rep, CRYPTO_NAMES, "Crypto regime")]
        if crypto_rep is not None else []
    )
    page = pagebuilder.render_page(
        "Morning Market Dashboard",
        generated,
        [pagebuilder.render_html_asof(asof),
         pagebuilder.render_html_headline(headline),
         pagebuilder.render_html_score(report, SCORE_NAMES),
         *crypto_blocks,
         pagebuilder.render_html_yield(yrep),
         pagebuilder.render_html_table(table, UNITS, WINDOWS, stale_after=3, labels=LABELS),
         pagebuilder.render_html_checks(check_text),
         pagebuilder.render_html_rules(rules.RULES)],
        notes,
    )
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {SITE_DIR / 'index.html'}")

if __name__ == "__main__":
    main()
