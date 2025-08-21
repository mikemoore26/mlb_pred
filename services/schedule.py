# services/schedule.py
from datetime import date
from utils.sources import get_schedule_range

def get_todays_schedule():
    today = date.today().isoformat()
    games = get_schedule_range(today, today)
    out = []
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "")
        home = ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("name", "")
        away = ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("name", "")
        tstr = (g.get("gameDate") or "")  # UTC ISO; format as needed in UI
        out.append({"home": home, "away": away, "time_str": tstr, "status": status})
    return out
