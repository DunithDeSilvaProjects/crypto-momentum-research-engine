def render_headline(trad: dict, crypto: dict | None, yrep: dict, names: dict | None = None) -> str:
    names = names or {}

    def top(items):
        if not items:
            return "nothing"
        key, value = items[0]
        return f"{names.get(key, key)} ({value:+.1f})"

    def regime(rep):
        return "unavailable" if rep is None else f"{rep['label']} ({rep['score']:+.1f})"

    lines = [f"Traditional regime {regime(trad)}; crypto regime {regime(crypto)}; yield regime {yrep['regime']}."]
    for title, rep in (("Traditional", trad), ("Crypto", crypto)):
        if rep is None:
            continue
        lines.append(f"{title}: lifted by {top(rep['drivers']['positive'])}, held back by {top(rep['drivers']['negative'])}.")
    return "\n".join(lines)