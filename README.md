## RoyalePredict

A Clash Royale deck matchup prediction and analytics service. It exposes:

- ML probability for deck A beating deck B using a trained, calibrated model
- Database-backed aggregate stats for top decks and historical head-to-head matchups
- Neighbor-based fallback estimates when exact matchups aren’t in the DB


### Features
- Antisymmetric feature engineering over decks; XGBoost + Platt calibration
- Time-aware train/test split (respects chronology)
- FastAPI service with endpoints for health, stats, matchups, and ML predictions
- Admin endpoints to refresh materialized views and hot-reload the model bundle


## Project overview and performance

### What this project is
- Predicts the probability that deck A beats deck B given two 8-card decks.
- Trains on historical matches, using antisymmetric features and XGBoost; outputs calibrated probabilities.
- Falls back to database-derived neighbor estimates and exact historical matchups when available.

### Model performance (current)
- Approximate test accuracy: **~60%** on a held-out, time-aware test split.
- Probabilities are calibrated via Platt scaling over a validation slice for better reliability.
- Note: accuracy is a coarse metric for probabilistic models; for interviews, mention calibration and AUC if available. Re-run training to capture fresh metrics.

### How to reproduce metrics
1. Ensure the DB has matches loaded (or set `TRAIN_SAMPLE_LIMIT` for a quick dry run).
2. Run:
```bash
python train_model.py
```
3. The script prints a metrics dict (train/test logloss, AUC, accuracy, and split info). Update the number above if your new run differs.


## Visualizations
You can generate ready-to-embed charts from your database (and optional model curves):

```bash
# Install plotting extras (already in requirements.txt)
pip install -r requirements.txt

# Generate charts from DB materialized views
python report.py --dsn "<your PG_DSN>" --out reports

# (Optional) Also compute ROC and calibration by training a quick model
python report.py --dsn "<your PG_DSN>" --out reports --with-model-curves --sample-limit 50000
```

This will create PNGs under `reports/`:
- `top_decks_winrate.png` — top decks by winrate (with min games filter)
- `top_decks_most_played.png` — most played decks
- `winrate_distribution.png` — distribution of deck winrates
- `roc_curve.png` — ROC on a held-out test split (optional)
- `calibration_curve.png` — probability calibration (optional)

You can drag-and-drop these into your README or GitHub release as needed.


## Prerequisites
- Python 3.11
- PostgreSQL (local or remote) with your base tables already created
- Supercell API token (optional, for collecting and refreshing cards/matches)


## Quickstart

1) Clone and enter the project folder
```bash
cd RoyalePredict
```

2) Create a virtual environment and install requirements
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Configure environment
```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```
Edit `.env` values:
- `PG_DSN` → your Postgres connection string
- `ADMIN_TOKEN` → any secret you’ll use to call admin endpoints
- `MODEL_PATH` → where the trained bundle is saved/loaded (default: `deck_model.pkl`)
- `SC_API_TOKEN` → Supercell API token (optional if not collecting)

4) Ensure materialized views exist
Your base tables are already created. Create the materialized views and indexes the API expects:
```bash
# Using psql
psql "<your PG_DSN without quotes>" -f db/materialized_views.sql
```

5) Load reference data (optional but recommended)
```bash
# Cards metadata from Supercell
python load_cards.py
```

6) Collect and load match data (optional if you already have matches)
```bash
# Collect high-ladder matches into ladder_only_matches.jsonl
python collect_high_ladder_debug.py

# Load JSONL into Postgres (de-duplicated by external hash)
python load_matches.py
```

7) Train the model (you said yours is already trained)
```bash
python train_model.py
```
This writes a calibrated model bundle to `MODEL_PATH` (default `deck_model.pkl`).

8) Serve the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```


## Configuration (.env)
See `.env.example` for all tunables. Common ones:
- `PG_DSN` (required by API and loaders)
- `ADMIN_TOKEN` (required for admin endpoints)
- `MODEL_PATH` (path to joblib bundle)
- `SC_API_TOKEN` (for `load_cards.py` and collector)


## Endpoints

All endpoints are served by `api.py`.

- Health
```text
GET /health
```
Response: service status, whether the ML model is loaded, vocab size, and neighbor defaults.

- Top decks by winrate
```text
GET /decks/top?limit=50&min_games=50
```

- Most played decks
```text
GET /decks/most_played?limit=50
```

- Resolve a deck ID from 8 card IDs (order-insensitive)
```text
POST /deck/resolve
Body: [26000000, 26000001, ... 8 ids]
```
If unknown, returns similar neighbors by card overlap.

- Exact matchup by deck IDs
```text
GET /matchup/{a_deck}/{b_deck}
```

- Matchup by cards (uses exact if present, else neighbor-weighted estimate)
```text
POST /matchup
Body: { "deckA": [8 ids], "deckB": [8 ids], "min_overlap": 7, "neighbors": 10 }
```

- ML prediction (cards + extra scalar features)
```text
POST /predict
Body: { "deckA": [8 ids], "deckB": [8 ids] }
```
Response includes `prob_a_wins`, unknown cards (not in vocab), and the extra features used.

- Admin: refresh materialized views
```text
POST /admin/refresh?concurrent=true
Header: X-Admin-Token: <ADMIN_TOKEN>
```

- Admin: hot-reload the model bundle
```text
POST /admin/reload_model
Header: X-Admin-Token: <ADMIN_TOKEN>
```


## Example requests (PowerShell)
```powershell
# Health
curl http://localhost:8000/health

# ML prediction
$body = @{ deckA = @(26000000,26000001,26000002,26000003,26000004,26000005,26000006,26000007); deckB = @(26000008,26000009,26000010,26000011,26000012,26000013,26000014,26000015) } | ConvertTo-Json
curl -Method POST -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $body

# Admin refresh (PowerShell)
curl -Method POST -Uri "http://localhost:8000/admin/refresh?concurrent=true" -Headers @{"X-Admin-Token"="<ADMIN_TOKEN>"}
```


## Data model (minimal)
- `cards(id, name, elixir_cost, rarity)`
- `players(tag, name, trophies, last_seen)`
- `decks(id, hash UNIQUE, card_ids int[])` (card IDs stored sorted; `hash` de-duplicates)
- `matches(..., a_deck_id, b_deck_id, a_won, ts, external_hash UNIQUE, ...)`

Materialized views created by `db/materialized_views.sql`:
- `deck_stats(deck_id, total_games, winrate)`
- `deck_stats_named(deck_id, card_ids, card_names, total_games, winrate)`
- `deck_matchups(a_deck_id, b_deck_id, games, a_winrate)`
- `deck_matchups_named(a_deck_id, a_cards, b_deck_id, b_cards, games, a_winrate)`


## Training notes
- Antisymmetric features: one-hot difference plus optional signed cross terms
- Time-aware split; best model picked by validation AUC; calibrated via Platt scaling
- Bundle includes model + vocab and metadata; API replicates vectorization


## Troubleshooting
- "ML model not loaded" → Ensure `MODEL_PATH` exists and `/admin/reload_model` returns ok
- "relation deck_stats_named does not exist" → Run `db/materialized_views.sql`, then hit `/admin/refresh`
- DB connection errors → verify `PG_DSN` in `.env` and that Postgres is reachable
- XGBoost GPU/CPU → set `XGB_DEVICE=cpu` if CUDA is not available
- Windows venv paths → use `venv\Scripts\python`, `venv\Scripts\uvicorn`




