import pandas as pd

from momentum import crypto


def test_parse_klines_keeps_finished_days_and_drops_the_unfinished_one():
    def row(open_ms, close, quote_volume):
        return [open_ms, "0", "0", "0", close, "0", open_ms + 86_399_999,
                quote_volume, 0, "0", "0", "0"]

    rows = [row(1_699_920_000_000, "105", "1050"),    # 14 Nov 2023
            row(1_700_006_400_000, "110", "2200"),    # 15 Nov 2023
            row(1_700_092_800_000, "112", "560")]     # 16 Nov 2023, still running
    now = pd.Timestamp("2023-11-16 12:00", tz="UTC")

    out = crypto.parse_klines(rows, now)

    assert list(out.index) == [pd.Timestamp("14 Nov 2023"), pd.Timestamp("15 Nov 2023")]
    assert out["close"].tolist() == [105, 110]
    assert out["quote_volume"].iloc[-1] == 2200
    assert out.index.tz is None