from momentum import snapshot
from momentum.macro import load_fred

UNITS = {
    "DGS10": "bp", "DGS2": "bp", "T10Y2Y": "bp", "DFII10": "bp", "T10YIE": "bp",
    "BAMLH0A0HYM2": "bp",
    "VIXCLS": "pct", "SP500": "pct", "NASDAQCOM": "pct",
    "DCOILWTICO": "pct", "DTWEXBGS": "pct",
}
WINDOWS = (1, 5, 20)


def main() -> None:
    df = load_fred(list(UNITS.keys()), start="2025-01-01")
    table = snapshot.snapshot_table(df, UNITS, windows=WINDOWS)
    print(snapshot.render_text(table, UNITS, windows=WINDOWS, stale_after=3))


if __name__ == "__main__":
    main()