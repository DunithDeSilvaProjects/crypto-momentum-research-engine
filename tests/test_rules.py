from momentum import rules


def test_every_rule_has_an_area_a_rule_and_a_known_status():
    assert len(rules.RULES) == 16
    for r in rules.RULES:
        assert set(r) == {"area", "rule", "status"}
        assert r["status"] in {"convention", "judgment", "proxy"}


def test_crypto_rules_quote_the_numbers_that_live_in_the_code():
    by_area = {r["area"]: r for r in rules.RULES}

    assert "trend 20" in by_area["Crypto components"]["rule"]
    assert "etf flows 10" in by_area["Crypto components"]["rule"]
    assert "70%" in by_area["Crypto minimum coverage"]["rule"]
    assert by_area["Crypto data"]["status"] == "proxy"