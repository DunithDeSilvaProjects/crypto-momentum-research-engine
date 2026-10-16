from momentum import snapshot
from momentum.macro import load_fred

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


def main() -> None:
    df = load_fred(list(UNITS.keys()), start="2025-01-01")
    table = snapshot.snapshot_table(df, UNITS, windows=WINDOWS)
    print(snapshot.render_text(table, UNITS, windows=WINDOWS, stale_after=3, labels=LABELS))


if __name__ == "__main__":
    main()