# collect_high_ladder_debug.py
import os, time, json, hashlib, sys
from typing import Dict, Any, List, Iterable, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("SC_API_TOKEN")
if not API:
    print("Missing SC_API_TOKEN in .env", file=sys.stderr)
    sys.exit(1)

BASE = "https://api.clashroyale.com/v1"
H    = {"Authorization": f"Bearer {API}"}

SEED_LOCATION_NAMES = ["Global", "United States", "Canada", "Japan", "Germany", "South Korea", "Brazil", "France", "Spain"]
TOP_CLANS_PER_LOCATION = 50
CLAN_MEMBERS_LIMIT = 50

# Expanded set so we can observe what’s coming back
ALLOWED_MODES = {
    "ladder", "pvp"
}

REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.25
SLEEP_ON_429 = 2.0
MAX_RETRIES = 5

OUT_PATH = "ladder_only_matches.jsonl"


# ---------- HTTP helpers ----------
def req(url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    backoff = SLEEP_ON_429
    for _ in range(MAX_RETRIES):
        r = requests.get(url, headers=H, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code in (429,) or (500 <= r.status_code < 600):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        return r
    r.raise_for_status()
    return r

def paged_get(url: str, per_page: int = 50, max_items: int = 200) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {"limit": per_page}
    while True:
        r = req(url, params)
        if not r.ok:
            break
        data = r.json()
        page_items = data.get("items", [])
        items.extend(page_items)
        if len(items) >= max_items:
            break
        after = data.get("paging", {}).get("cursors", {}).get("after")
        if not after:
            break
        params = {"limit": per_page, "after": after}
        time.sleep(SLEEP_BETWEEN_CALLS)
    return items


# ---------- API wrappers ----------
def list_locations() -> List[Dict[str, Any]]:
    return paged_get(f"{BASE}/locations", per_page=200, max_items=1000)

def find_location_ids() -> Dict[str, int]:
    all_locs = list_locations()
    result: Dict[str, int] = {}
    for want in SEED_LOCATION_NAMES:
        lw = want.lower()
        m = next((loc for loc in all_locs if lw in loc["name"].lower()), None)
        if m:
            result[want] = m["id"]
    if "Global" not in result:
        result["Global"] = 57000000
    return result

def top_clans(location_id: int, target: int) -> List[Dict[str, Any]]:
    return paged_get(f"{BASE}/locations/{location_id}/rankings/clans", per_page=50, max_items=target)

def clan_members(clan_tag: str) -> List[Dict[str, Any]]:
    r = req(f"{BASE}/clans/%23{clan_tag.strip('#')}/members", {"limit": CLAN_MEMBERS_LIMIT})
    if not r.ok: return []
    return r.json().get("items", [])

def player_profile(player_tag: str) -> Dict[str, Any]:
    r = req(f"{BASE}/players/%23{player_tag.strip('#')}")
    r.raise_for_status()
    return r.json()

def player_battlelog(player_tag: str) -> List[Dict[str, Any]]:
    r = req(f"{BASE}/players/%23{player_tag.strip('#')}/battlelog")
    if not r.ok: return []
    return r.json()


# ---------- Normalize ----------
def row_hash(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()

def normalize_match(b: Dict[str, Any]) -> Dict[str, Any]:
    team = (b.get("team") or [{}])[0]
    opp  = (b.get("opponent") or [{}])[0]
    a_cards = [c.get("id") for c in (team.get("cards") or [])]
    b_cards = [c.get("id") for c in (opp.get("cards") or [])]
    a_crowns = team.get("crowns", 0)
    b_crowns = opp.get("crowns", 0)
    result = 1 if a_crowns > b_crowns else (0 if a_crowns < b_crowns else None)
    rec = {
        "battleTime": b.get("battleTime"),
        "mode": (b.get("gameMode") or {}).get("name"),
        "teamTag": team.get("tag"),
        "oppTag": opp.get("tag"),
        "teamDeck": a_cards,
        "oppDeck": b_cards,
        "result": result,
    }
    rec["hash"] = row_hash(rec)
    return rec


# ---------- Seeding ----------
def iter_seed_player_tags() -> Iterable[str]:
    loc_ids = find_location_ids()
    seen_clans = set()
    for loc_name, loc_id in loc_ids.items():
        clans = top_clans(loc_id, TOP_CLANS_PER_LOCATION)
        for c in clans:
            ctag = c.get("tag")
            if not ctag or ctag in seen_clans:
                continue
            seen_clans.add(ctag)
            members = clan_members(ctag)
            for m in members:
                ptag = m.get("tag")
                if ptag:
                    yield ptag
            time.sleep(SLEEP_BETWEEN_CALLS)


# ---------- Main collect ----------
def collect_matches(max_players: int = 10, max_matches_per_player: int = 25) -> None:
    out_seen = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    h = json.loads(line).get("hash")
                    if h: out_seen.add(h)
                except json.JSONDecodeError:
                    pass

    written = 0
    checked_players = 0
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for tag in iter_seed_player_tags():
            if checked_players >= max_players:
                break
            checked_players += 1

            try:
                prof = player_profile(tag)
            except Exception:
                time.sleep(SLEEP_BETWEEN_CALLS); continue

            trophies_now = prof.get("trophies") or 0

            try:
                logs = player_battlelog(tag)
            except Exception:
                time.sleep(SLEEP_BETWEEN_CALLS); continue

            kept = 0
            seen = 0
            for b in logs:
                mode_name = ((b.get("gameMode") or {}).get("name") or "").lower()
                if mode_name not in ALLOWED_MODES:
                    continue
                seen += 1

                if (len((b.get("team") or [{}])[0].get("cards") or []) != 8 or
                    len((b.get("opponent") or [{}])[0].get("cards") or []) != 8):
                    continue
                if (b.get("team") or [{}])[0].get("crowns") is None:
                    continue

                rec = normalize_match(b)
                h = rec["hash"]
                if h in out_seen:
                    continue

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_seen.add(h)
                written += 1
                kept += 1
                if kept >= max_matches_per_player:
                    break

            print(f"[{checked_players}] {prof.get('name','?')} {tag} "
                  f"(now {trophies_now}) -> seen {seen}, kept {kept}")

            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[summary] checked_players={checked_players}, wrote={written}, unique_hashes={len(out_seen)}")


if __name__ == "__main__":
    print("Seeding (expanded modes, no trophy filter)…")
    collect_matches(max_players=15000, max_matches_per_player=25)
