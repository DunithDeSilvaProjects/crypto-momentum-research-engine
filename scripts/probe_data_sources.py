"""Diagnostic: which public data sources answer from this machine? Prints status codes only."""
import time

import requests

ENDPOINTS = {
    "binance futures klines": "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=2",
    "binance futures funding": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=2",
    "binance futures open interest": "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
    "binance spot (public data host)": "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2",
    "bybit klines": "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=D&limit=2",
    "okx klines": "https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=1D&limit=2",
    "coingecko ping": "https://api.coingecko.com/api/v3/ping",
    "defillama stablecoins": "https://stablecoins.llama.fi/stablecoins?includePrices=false",
}


def probe(name: str, url: str) -> str:
    start = time.time()
    try:
        r = requests.get(url, timeout=30, stream=True)
        status = r.status_code
        snippet = r.raw.read(80, decode_content=True).decode("utf-8", "replace").replace("\n", " ")
        r.close()
    except requests.RequestException as exc:
        return f"{name:34s} ERROR {type(exc).__name__}"
    return f"{name:34s} {status}  {time.time() - start:.1f}s  {snippet}"


if __name__ == "__main__":
    for name, url in ENDPOINTS.items():
        print(probe(name, url))