from momentum import brief

NAMES = {"real_yield": "10y real yield", "spreads": "HY credit spreads",
         "breadth": "Market breadth", "stablecoins": "Stablecoin supply"}

TRAD = {"score": 8.3, "label": "neutral",
        "drivers": {"positive": [("real_yield", 20.0), ("equities", 4.2)],
                    "negative": [("spreads", -14.2), ("curve", -4.5)]}}
CRYPTO = {"score": 1.4, "label": "neutral",
          "drivers": {"positive": [("breadth", 6.0)], "negative": [("stablecoins", -6.4)]}}


def test_render_headline_names_the_regimes_and_the_biggest_driver_each_way():
    lines = brief.render_headline(TRAD, CRYPTO, {"regime": "mixed"}, NAMES).split("\n")

    assert lines == [
        "Traditional regime neutral (+8.3); crypto regime neutral (+1.4); yield regime mixed.",
        "Traditional: lifted by 10y real yield (+20.0), held back by HY credit spreads (-14.2).",
        "Crypto: lifted by Market breadth (+6.0), held back by Stablecoin supply (-6.4).",
    ]


def test_render_headline_handles_a_missing_crypto_report_and_empty_drivers():
    empty = {"score": 0.0, "label": "neutral", "drivers": {"positive": [], "negative": []}}

    lines = brief.render_headline(empty, None, {"regime": "mixed"}, NAMES).split("\n")

    assert lines == [
        "Traditional regime neutral (+0.0); crypto regime unavailable; yield regime mixed.",
        "Traditional: lifted by nothing, held back by nothing.",
    ]