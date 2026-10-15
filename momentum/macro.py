import pandas as pd
import requests
import os
from pathlib import Path

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.Series:
    """Daily observations of a FRED series as floats; FRED's '.' (no data) becomes NaN."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "frequency": "d",
        "observation_start": start,
    }
    resp = requests.get(FRED_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    values = pd.to_numeric([o["value"] for o in obs], errors="coerce")  # '.' -> NaN
    return pd.Series(values, index=dates, name=series_id)

def fred_api_key() -> str:
    """Read the FRED key from the environment; fail loudly if it is missing."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("The FRED_API_KEY environment variable is not set")
    return key

def load_fred(series_ids: list[str], start: str) -> pd.DataFrame:
    """One column per FRED series, rows aligned by date (a date missing from a series is NaN)."""
    key = fred_api_key()
    series = [fetch_fred_series(sid, key, start) for sid in series_ids]
    return pd.concat(series, axis=1).sort_index()

def get_fred_series(series_id: str, start: str, cache_dir: str = "data_cache") -> pd.Series:
    """FRED series cached as Parquet; re-fetches the last week so late or revised values are picked up."""
    path = Path(cache_dir) / f"fred_{series_id}.parquet"
    key = fred_api_key()
    if path.exists():
        old = pd.read_parquet(path)[series_id]
        fetch_from = (old.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        new = fetch_fred_series(series_id, key, fetch_from)
        out = pd.concat([old, new])
    else:
        out = fetch_fred_series(series_id, key, start)

    out = out[~out.index.duplicated(keep="last")]  # keep the last value if a date was re-fetched
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_frame().to_parquet(path)
    return out