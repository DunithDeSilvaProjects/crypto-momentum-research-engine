import pandas as pd
import momentum.data as data


def make_df(days):
    idx = pd.date_range(days[0], days[-1], freq="D", tz="UTC")
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


def test_first_call_downloads_and_writes_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_history(symbol, start, interval="1d"):
        calls.append(start)
        return make_df(["2024-01-01", "2024-01-05"])

    monkeypatch.setattr(data, "fetch_history", fake_fetch_history)

    df = data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))

    assert len(df) == 5            # 5 days, Jan 1 to Jan 5
    assert calls == ["2024-01-01"]
    assert (tmp_path / "BTCUSDT_1d.parquet").exists()


def test_second_call_fetches_only_after_last_cached_day(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_history(symbol, start, interval="1d"):
        calls.append(start)
        if start == "2024-01-01":
            return make_df(["2024-01-01", "2024-01-05"])
        return make_df(["2024-01-06", "2024-01-07"])

    monkeypatch.setattr(data, "fetch_history", fake_fetch_history)

    data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))
    df = data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))

    assert calls == ["2024-01-01", "2024-01-06"]   # the day after Jan 5
    assert len(df) == 7
    assert df.index.is_unique


def test_overlapping_rows_are_not_duplicated(tmp_path, monkeypatch):
    def fake_fetch_history(symbol, start, interval="1d"):
        if start == "2024-01-01":
            return make_df(["2024-01-01", "2024-01-05"])
        return make_df(["2024-01-05", "2024-01-07"])   # Jan 5 again

    monkeypatch.setattr(data, "fetch_history", fake_fetch_history)

    data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))
    df = data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))

    assert df.index.is_unique
    assert len(df) == 7

'''
Add a fourth test, test_up_to_date_cache_returns_unchanged. The fake fetch_history 
returns pd.DataFrame() for the second call. Assert that len(df) stays 5 and that 
the index is still a UTC DatetimeIndex. Predict first: does it pass today, and why 
or why not? If it fails, fix get_history, with one hint: a guard like 
if not new.empty.
'''
def  test_up_to_date_cache_returns_unchanged(tmp_path, monkeypatch):
    def fake_fetch_history(symbol, start, interval="1d"):
        if start == "2024-01-01":
            return make_df(["2024-01-01", "2024-01-05"])
        return pd.DataFrame()   # no new data

    monkeypatch.setattr(data, "fetch_history", fake_fetch_history)

    data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))
    df = data.get_history("BTCUSDT", "2024-01-01", cache_dir=str(tmp_path))

    assert len(df) == 5
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df["close"].tolist() == [0, 1, 2, 3, 4]