# load_cards.py
import os
import sys
import requests
import psycopg2
from dotenv import load_dotenv
from typing import List, Tuple, Any
from psycopg2.extras import execute_values

load_dotenv()

API = os.getenv("SC_API_TOKEN")
DSN = os.getenv("PG_DSN")

if not API:
    print("Missing SC_API_TOKEN in .env", file=sys.stderr)
    sys.exit(1)
if not DSN:
    print("Missing PG_DSN in .env", file=sys.stderr)
    sys.exit(1)

BASE = "https://api.clashroyale.com/v1"
H    = {"Authorization": f"Bearer {API}"}

def ensure_schema(conn) -> None:
    """Add columns if they don't exist yet."""
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE cards ADD COLUMN IF NOT EXISTS elixir_cost INT;")
        cur.execute("ALTER TABLE cards ADD COLUMN IF NOT EXISTS rarity TEXT;")
    conn.commit()

def fetch_cards() -> List[Tuple[Any, ...]]:
    """Fetch id, name, elixir_cost, rarity from Supercell API."""
    r = requests.get(f"{BASE}/cards", headers=H, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", []) or []

    rows = []
    for it in items:
        cid = int(it.get("id"))
        name = it.get("name") or ""
        elixir = it.get("elixirCost")
        try:
            elixir_cost = int(elixir) if elixir is not None else 0
        except (TypeError, ValueError):
            elixir_cost = 0
        rarity = (it.get("rarity") or "unknown").lower()
        rows.append((cid, name, elixir_cost, rarity))
    return rows

def upsert_cards(rows: List[Tuple[Any, ...]]) -> None:
    sql = """
        INSERT INTO cards (id, name, elixir_cost, rarity)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET name = EXCLUDED.name,
            elixir_cost = EXCLUDED.elixir_cost,
            rarity = EXCLUDED.rarity;
    """
    with psycopg2.connect(DSN) as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()
    print(f"Upserted {len(rows)} cards.")

if __name__ == "__main__":
    rows = fetch_cards()
    if not rows:
        print("No cards returned from API.", file=sys.stderr)
        sys.exit(2)
    upsert_cards(rows)
