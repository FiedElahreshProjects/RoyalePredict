# sanity_check.py
import json, collections, datetime

PATH = "ladder_only_matches.jsonl"   # <-- matches your collector

modes = collections.Counter()
total = 0
bad_json = 0
no_decks = 0
no_result = 0
dupes = 0
hashes = set()
players = set()
deck_sizes = collections.Counter()
date_min = None
date_max = None

def parse_ts(s: str) -> datetime.datetime:
    # battleTime example: "20250907T210143.000Z"
    return datetime.datetime.strptime(s, "%Y%m%dT%H%M%S.%fZ")

with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad_json += 1
            continue

        # dedupe check
        h = r.get("hash")
        if h in hashes:
            dupes += 1
            continue
        if h:
            hashes.add(h)

        # mode
        modes[(r.get("mode") or "").lower()] += 1

        # date span
        bt = r.get("battleTime")
        if bt:
            ts = parse_ts(bt)
            if not date_min or ts < date_min: date_min = ts
            if not date_max or ts > date_max: date_max = ts

        # players seen
        a_tag = r.get("teamTag")
        b_tag = r.get("oppTag")
        if a_tag: players.add(a_tag)
        if b_tag: players.add(b_tag)

        # deck completeness
        a_deck = r.get("teamDeck") or []
        b_deck = r.get("oppDeck") or []
        deck_sizes[len(a_deck)] += 1
        deck_sizes[len(b_deck)] += 1
        if len(a_deck) != 8 or len(b_deck) != 8:
            no_decks += 1

        # result presence
        if r.get("result") is None:
            no_result += 1

print("=== Sanity Report ===")
print(f"File: {PATH}")
print(f"Total lines:          {total}")
print(f"JSON errors:          {bad_json}")
print(f"Duplicate hashes:     {dupes}")
print(f"Unique matches:       {len(hashes)}")
print(f"Unique players:       {len(players)}")
print(f"Modes:                {dict(modes)}")
if date_min and date_max:
    print(f"Date span:            {date_min}  ->  {date_max}")
print(f"Deck size counts:     {dict(deck_sizes)}")
print(f"Records w/ !8-card:   {no_decks}")
print(f"Records w/ no result: {no_result}")
