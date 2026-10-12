from dataclasses import dataclass

ANN = 365   # crypto trades every day, so 365 trading days a year

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT",
    "UNIUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "AAVEUSDT", "ARBUSDT",
)

@dataclass(frozen=True)
class SignalConfig:
    lookbacks: tuple = (30, 60, 90)
    vol_halflife: int = 30
    winsor: float = 3.0