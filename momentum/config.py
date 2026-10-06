from dataclasses import dataclass

ANN = 365   # crypto trades every day, so 365 trading days a year

@dataclass(frozen=True)
class SignalConfig:
    lookbacks: tuple = (30, 60, 90)
    vol_halflife: int = 30
    winsor: float = 3.0