from momentum import rules


def test_every_rule_has_an_area_a_rule_and_a_known_status():
    assert len(rules.RULES) == 11
    for r in rules.RULES:
        assert set(r) == {"area", "rule", "status"}
        assert r["status"] in {"convention", "judgment", "proxy"}