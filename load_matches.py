# load_matches.py
import os, json, hashlib, datetime
from typing import Iterable, List, Tuple, Optional, Dict, Any
import psycopg2
from psycopg2.extras import execute_values

DSN   = os.getenv("PG_DSN", "postgresql://postgres:Feeda123@localhost:5432/Clash")
INPUT = os.getenv("INPUT_JSONL", "ladder_only_matches.jsonl")

# ---------- Helpers ----------
def deck_hash(card_ids: Iterable[int]) -> str:
    s = ",".join(str(int(c)) for c in sorted(card_ids))
    return hashlib.sha1(s.encode()).hexdigest()

def parse_ts(s: str) -> datetime.datetime:
    """
    Handles Clash Royale battleTime like:
      '20250701T095506.000Z' or '20250701T095506Z'
    Returns a timezone-aware UTC datetime.
    """
    s = s.strip()
    fmts = ["%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"]
    for fmt in fmts:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            # attach UTC tzinfo
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized battleTime format: {s}")

# ---------- One-time schema safety (idempotent) ----------
DDL = """
-- players.tag should be unique (or PK)
CREATE UNIQUE INDEX IF NOT EXISTS players_tag_uidx ON players(tag);

-- decks.hash should be unique
CREATE UNIQUE INDEX IF NOT EXISTS decks_hash_uidx ON decks(hash);

-- matches.external_hash for de-dupe across loads
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS external_hash text;

CREATE UNIQUE INDEX IF NOT EXISTS matches_external_hash_uidx
ON matches(external_hash);

-- helpful composite index if you still use it as a fallback
CREATE INDEX IF NOT EXISTS matches_ab_ts_mode_idx
ON matches(a_player_tag, b_player_tag, ts, mode);
"""

# ---------- Upserts ----------
def ensure_player(cur, tag: Optional[str]) -> None:
    if not tag:
        return
    cur.execute(
        """
        INSERT INTO players (tag, name, trophies, last_seen)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (tag) DO UPDATE
        SET last_seen = EXCLUDED.last_seen
        """,
        (tag, None, None),
    )

def ensure_deck_id(cur, card_ids: Iterable[int]) -> int:
    cards_sorted = [int(c) for c in sorted(card_ids)]
    h = deck_hash(cards_sorted)
    cur.execute("SELECT id FROM decks WHERE hash = %s", (h,))
    row = cur.fetchone()
    if row:
        return row[0]
    # store sorted card_ids to keep order-insensitive identity everywhere
    cur.execute(
        """
        INSERT INTO decks (hash, card_ids)
        VALUES (%s, %s)
        RETURNING id
        """,
        (h, cards_sorted),
    )
    return cur.fetchone()[0]

# ---------- Loader ----------
def load(batch_size: int = 1000) -> None:
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # One-time DDL (idempotent)
    cur.execute(DDL)
    conn.commit()

    to_insert: List[Tuple[Any, ...]] = []
    total = 0
    skipped_noresult = 0
    skipped_dupe = 0

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            # Skip missing results (ties / invalid)
            res = rec.get("result", None)
            if res is None:
                skipped_noresult += 1
                continue

            # De-dupe using the JSONL hash
            ext_hash = rec.get("hash")
            if not ext_hash:
                # fallback: derive from core fields
                ext_hash = hashlib.sha1(
                    json.dumps(
                        {
                            "t": rec.get("battleTime"),
                            "m": (rec.get("mode") or "").lower(),
                            "ta": rec.get("teamTag"),
                            "tb": rec.get("oppTag"),
                            "da": sorted(rec.get("teamDeck") or []),
                            "db": sorted(rec.get("oppDeck") or []),
                            "r": int(res),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()

            # quick existence check (avoid expensive work if duplicate)
            cur.execute("SELECT 1 FROM matches WHERE external_hash = %s", (ext_hash,))
            if cur.fetchone():
                skipped_dupe += 1
                continue

            try:
                ts = parse_ts(rec["battleTime"])
            except Exception:
                # skip weird timestamps
                continue

            mode = (rec.get("mode") or "Unknown").lower()
            tier_bin = "high_ladder"

            a_tag = rec.get("teamTag")
            b_tag = rec.get("oppTag")
            a_deck = rec.get("teamDeck") or []
            b_deck = rec.get("oppDeck") or []

            # Upsert players now (lightweight)
            ensure_player(cur, a_tag)
            ensure_player(cur, b_tag)

            # Resolve deck ids (sorted card_ids persisted)
            a_deck_id = ensure_deck_id(cur, a_deck)
            b_deck_id = ensure_deck_id(cur, b_deck)

            a_won = bool(res)

            to_insert.append(
                (
                    ts,
                    mode,
                    tier_bin,
                    a_tag,
                    b_tag,
                    a_deck_id,
                    b_deck_id,
                    a_won,
                    None,          # trophy_change, unknown here
                    "supercell",   # source
                    ext_hash,      # external_hash for de-dupe
                )
            )

            if len(to_insert) >= batch_size:
                execute_values(
                    cur,
                    """
                    INSERT INTO matches (
                        ts, mode, tier_bin,
                        a_player_tag, b_player_tag,
                        a_deck_id, b_deck_id,
                        a_won, trophy_change, source,
                        external_hash
                    )
                    VALUES %s
                    ON CONFLICT (external_hash) DO NOTHING
                    """,
                    to_insert,
                )
                conn.commit()
                total += len(to_insert)
                to_insert.clear()

    # flush remaining
    if to_insert:
        execute_values(
            cur,
            """
            INSERT INTO matches (
                ts, mode, tier_bin,
                a_player_tag, b_player_tag,
                a_deck_id, b_deck_id,
                a_won, trophy_change, source,
                external_hash
            )
            VALUES %s
            ON CONFLICT (external_hash) DO NOTHING
            """,
            to_insert,
        )
        conn.commit()
        total += len(to_insert)

    cur.close()
    conn.close()
    print(f"✅ Done. Inserted ~{total} new matches. Skipped (no result): {skipped_noresult}, skipped (dupe): {skipped_dupe}")

if __name__ == "__main__":
    load(batch_size=2000)
