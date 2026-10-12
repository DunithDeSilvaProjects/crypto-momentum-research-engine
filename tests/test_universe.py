from momentum.config import DEFAULT_SYMBOLS


def test_universe_is_20_unique_usdt_perps():
    assert isinstance(DEFAULT_SYMBOLS, tuple)
    assert len(DEFAULT_SYMBOLS) == 20
    assert len(set(DEFAULT_SYMBOLS)) == len(DEFAULT_SYMBOLS)   # no duplicates
    assert all(s.endswith("USDT") for s in DEFAULT_SYMBOLS)