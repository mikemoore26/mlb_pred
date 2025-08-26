def cc_wpct_season() -> Dict[str, float]:
    k = cache._make_key("wpct_season", "cur")
    cached = cache.load_json("wpct_season", k, max_age_days=1)
    if cached is not None:
        return cached
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}"
    data = _safe_get_json(url) or {}
    wpct: Dict[str, float] = {}
    for rec in data.get("records", []):
        for t in rec.get("teamRecords", []):
            wins = t.get("wins", 0)
            losses = t.get("losses", 0)
            name = t.get("team", {}).get("name")
            if not name:
                continue
            wpct[name] = (wins / (wins + losses)) if (wins + losses) > 0 else 0.5
    cache.save_json("wpct_season", wpct, k)
    return wpct

def cc_wpct_last30() -> Dict[str, float]:
    k = cache._make_key("wpct_last30", "global")
    cached = cache.load_json("wpct_last30", k, max_age_days=1)
    if cached is not None:
        return cached
    end_date = _dt.date.today()
    start_date = end_date - _dt.timedelta(days=30)
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1"
    data = _safe_get_json(url) or {}
    counts: Dict[str, Dict[str, int]] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if (g.get("status", {}).get("detailedState") or "").lower() != "final":
                continue
            ht = g["teams"]["home"]["team"]["name"]
            at = g["teams"]["away"]["team"]["name"]
            hs = g["teams"]["home"].get("score", 0)
            as_ = g["teams"]["away"].get("score", 0)
            winner = ht if hs > as_ else at
            for t in (ht, at):
                counts.setdefault(t, {"wins": 0, "games": 0})
                counts[t]["games"] += 1
                if t == winner:
                    counts[t]["wins"] += 1
    result = {t: (v["wins"] / v["games"]) if v["games"] else 0.5 for t, v in counts.items()}
    cache.save_json("wpct_last30", result, k)
    return result
