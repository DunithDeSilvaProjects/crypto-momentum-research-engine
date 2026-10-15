import pandas as pd
import requests
import os

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