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
    "okx funding history": "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=3",
    "okx open interest now": "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP",
    "okx open interest history": "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy=BTC&period=1D",
    "binance spot 1000 days (ETH)": "https://data-api.binance.vision/api/v3/klines?symbol=ETHUSDT&interval=1d&limit=1000",
    "coingecko global": "https://api.coingecko.com/api/v3/global",
    "coingecko top 20": "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=20&page=1",
    "defillama stablecoin history": "https://stablecoins.llama.fi/stablecoincharts/all",
        "okx liquidations": "https://www.okx.com/api/v5/public/liquidation-orders?instType=SWAP&uly=BTC-USDT&state=filled&limit=3",
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