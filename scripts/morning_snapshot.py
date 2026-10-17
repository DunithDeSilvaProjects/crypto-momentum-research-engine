from momentum import snapshot
from momentum.macro import load_fred
from momentum import scoring

UNITS = {
    "DGS10": "bp", "DGS2": "bp", "T10Y2Y": "bp", "DFII10": "bp", "T10YIE": "bp",
    "BAMLH0A0HYM2": "bp",
    "VIXCLS": "pct", "SP500": "pct", "NASDAQCOM": "pct",
    "DCOILWTICO": "pct", "DTWEXBGS": "pct",
}
WINDOWS = (1, 5, 20)

LABELS = {
    "DGS10": "US 10y yield", "DGS2": "US 2y yield", "T10Y2Y": "2s10s curve",
    "DFII10": "10y real yield", "T10YIE": "10y breakeven", "BAMLH0A0HYM2": "HY credit spread",
    "VIXCLS": "VIX", "SP500": "S&P 500", "NASDAQCOM": "Nasdaq Composite",
    "DCOILWTICO": "WTI crude", "DTWEXBGS": "USD broad index",
}
TRADITIONAL_SPEC = {
    "equities":   {"col": "SP500",        "unit": "pct", "up_is_good": True,  "weight": 20},
    "vix":        {"col": "VIXCLS",       "unit": "pct", "up_is_good": False, "weight": 15},
    "spreads":    {"col": "BAMLH0A0HYM2", "unit": "bp",  "up_is_good": False, "weight": 15},
    "dollar":     {"col": "DTWEXBGS",     "unit": "pct", "up_is_good": False, "weight": 15},
    "real_yield": {"col": "DFII10",       "unit": "bp",  "up_is_good": False, "weight": 20},
    "curve":      {"col": "T10Y2Y",       "unit": "bp",  "up_is_good": True,  "weight": 15},
}
SCORE_NAMES = {
    "equities": "Equity trend", "vix": "VIX", "spreads": "HY credit spreads",
    "dollar": "Dollar (broad)", "real_yield": "10y real yield", "curve": "10y-2y curve",
}

MAX_AGE_DAYS = 3

assert sum(c["weight"] for c in TRADITIONAL_SPEC.values()) == 100

def main() -> None:
    df = load_fred(list(UNITS.keys()), start="2025-01-01")
    table = snapshot.snapshot_table(df, UNITS, windows=WINDOWS)
    print(snapshot.render_text(table, UNITS, windows=WINDOWS, stale_after=3, labels=LABELS))
    stale = scoring.stale_columns(df, MAX_AGE_DAYS)
    report = scoring.traditional_report(df.drop(columns=stale), TRADITIONAL_SPEC, n=5, lookback=60)
    print()
    print(scoring.render_score(report, SCORE_NAMES))
    if stale:
        print("Excluded as stale: " + ", ".join(LABELS.get(c, c) for c in stale))


if __name__ == "__main__":
    main()