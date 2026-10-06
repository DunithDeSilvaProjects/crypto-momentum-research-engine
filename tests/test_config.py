from dataclasses import dataclass
from dataclasses import FrozenInstanceError
from momentum.config import SignalConfig
import pytest


def test_defaults():
    cfg = SignalConfig()
    assert cfg.lookbacks == (30, 60, 90)
    assert cfg.vol_halflife == 30
    assert cfg.winsor == 3.0

def test_config_is_frozen():
    cfg = SignalConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.vol_halflife = 10
        
   