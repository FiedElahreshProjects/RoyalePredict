# api.py
import os
from typing import List, Dict, Any, Optional, Iterable, Tuple

import asyncpg
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, ClassifierMixin

load_dotenv()

DB_DSN = os.getenv("PG_DSN", "postgresql://postgres:Feeda123@localhost:5432/Clash")
ADMIN_TOKEN: Optional[str] = os.getenv("ADMIN_TOKEN")  # set in .env, e.g. ADMIN_TOKEN=supersecret
MODEL_PATH = os.getenv("MODEL_PATH", "deck_model.pkl")

app = FastAPI(title="RoyalePredict API", version="0.4.0")

# ----------------------------------------------------------------------
# Helpers copied from training so the pickle can resolve custom classes
# ----------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

class _PrefitXGBClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a prefit xgboost Booster."""
    def __init__(self, booster=None, best_iter=0):
        self.booster = booster
        self.best_iter = int(best_iter)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        return self  # prefit

    def predict_proba(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        p = self.booster.predict(d, iteration_range=(0, self.best_iter + 1))
        return np.vstack([1.0 - p, p]).T

class _PlattCalibrated(BaseEstimator, ClassifierMixin):
    """Platt scaling (logistic) on top of a prefit XGB wrapper."""
    def __init__(self, base=None, platt=None):
        self.base = base
        self.platt = platt
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        return self  # prefit

    def predict_proba(self, X):
        import xgboost as xgb
        raw = self.base.booster.predict(
            xgb.DMatrix(X),
            iteration_range=(0, self.base.best_iter + 1),
        )
        feat = _logit(raw).reshape(-1, 1)
        p1 = self.platt.predict_proba(feat)[:, 1]
        return np.vstack([1 - p1, p1]).T

# ----------------------------
# DB pool
# ----------------------------
async def get_pool():
    if not hasattr(app.state, "pool"):
        app.state.pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=10)
    return app.state.pool

# ----------------------------
# Helpers
# ----------------------------
def normalize_cards(cards: List[int]) -> List[int]:
    """Sort to make deck identity order-insensitive."""
    return sorted(int(c) for c in cards)

def require_admin(x_admin_token: Optional[str]):
    if not ADMIN_TOKEN:
        raise HTTPException(500, "ADMIN_TOKEN not configured on server.")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized.")

def _deck_key(cards: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(c) for c in cards))

# ----------------------------
# ML model loader + vectorizers
# ----------------------------
def _load_model_into_state() -> bool:
    # default “empty” state
    app.state.model = None
    app.state.vocab = []
    app.state.card_to_idx = {}
    app.state.extra_names = []
    app.state.priors = {}
    app.state.neigh_min_ov = 7
    app.state.neigh_k = 10

    if not os.path.exists(MODEL_PATH):
        return False

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    vocab = bundle["vocab"]
    card_to_idx = {cid: i for i, cid in enumerate(vocab)}

    # Optional extras from the upgraded training script
    extra_names = bundle.get("extra_feature_names", [])
    raw_priors = bundle.get("train_priors", {})  # {deck_key(tuple/list): {games,wins,winrate,pickrate}}
    priors: Dict[Tuple[int, ...], Dict[str, float]] = {}
    for k, v in raw_priors.items():
        # keys may come back as tuple or list; normalize into tuple[int,...]
        if isinstance(k, tuple):
            tkey = tuple(int(x) for x in k)
        else:
            try:
                tkey = tuple(int(x) for x in k)
            except TypeError:
                tkey = tuple([int(x) for x in list(k)])
        priors[tkey] = {
            "games": float(v.get("games", 0)),
            "wins": float(v.get("wins", 0)),
            "winrate": float(v.get("winrate", 0.5)),
            "pickrate": float(v.get("pickrate", 0.0)),
        }

    app.state.model = model
    app.state.vocab = vocab
    app.state.card_to_idx = card_to_idx
    app.state.extra_names = list(extra_names)
    app.state.priors = priors
    app.state.neigh_min_ov = int(bundle.get("neighbor_min_overlap", 7))
    app.state.neigh_k = int(bundle.get("neighbor_k", 10))
    return True

def _vec_cards_only(deckA: List[int], deckB: List[int]) -> (np.ndarray, List[int]):
    """
    Build +/-1 vector over vocab. Returns (vector, unknown_cards).
    Unknown cards are ignored in the vector but reported.
    """
    if not app.state.card_to_idx:
        raise HTTPException(503, "ML model not loaded.")
    d = len(app.state.vocab)
    x = np.zeros((1, d), dtype=np.float32)
    unknown = []
    for c in deckA:
        j = app.state.card_to_idx.get(c)
        if j is None: unknown.append(c)
        else: x[0, j] = 1.0
    for c in deckB:
        j = app.state.card_to_idx.get(c)
        if j is None: unknown.append(c)
        else: x[0, j] = -1.0
    return x, unknown

def _neighbor_backoff_prior(qkey: Tuple[int, ...]) -> (float, float):
    """
    Estimate (winrate, pickrate) for unknown deck by nearest neighbors
    from training priors using overlap >= neigh_min_ov.
    Weight by (overlap - min + 1)^2.
    """
    pri: Dict[Tuple[int, ...], Dict[str, float]] = app.state.priors or {}
    if not pri:
        return 0.5, 0.0
    qset = set(qkey)
    cands = []
    for key, p in pri.items():
        ov = len(qset.intersection(key))
        if ov >= app.state.neigh_min_ov:
            cands.append((ov, p))
    if not cands:
        return 0.5, 0.0
    cands.sort(key=lambda t: (-t[0], -t[1].get("games", 0.0)))
    cands = cands[: app.state.neigh_k]
    num_w = num_p = den = 0.0
    for ov, p in cands:
        w = float((ov - app.state.neigh_min_ov + 1) ** 2)
        num_w += w * float(p.get("winrate", 0.5))
        num_p += w * float(p.get("pickrate", 0.0))
        den   += w
    if den <= 0:
        return 0.5, 0.0
    return num_w / den, num_p / den

def _extra_feats(deckA: List[int], deckB: List[int]) -> Dict[str, float]:
    """
    Build the 5 scalar features the model was trained with:
      A_prior_win, B_prior_win, A_pickrate, B_pickrate, overlap_norm
    Priors come from TRAIN priors with neighbor backoff.
    """
    ak = _deck_key(deckA)
    bk = _deck_key(deckB)
    pri = app.state.priors or {}
    A = pri.get(ak)
    B = pri.get(bk)
    if A is None:
        a_wr, a_pr = _neighbor_backoff_prior(ak)
    else:
        a_wr, a_pr = float(A["winrate"]), float(A["pickrate"])
    if B is None:
        b_wr, b_pr = _neighbor_backoff_prior(bk)
    else:
        b_wr, b_pr = float(B["winrate"]), float(B["pickrate"])
    overlap = len(set(ak).intersection(bk)) / 8.0
    return {
        "A_prior_win": a_wr,
        "B_prior_win": b_wr,
        "A_pickrate": a_pr,
        "B_pickrate": b_pr,
        "overlap_norm": overlap,
    }

def _vec_full(deckA: List[int], deckB: List[int]) -> (np.ndarray, List[int], Dict[str, float]):
    """
    Concatenate card vector and extra scalar features in the same order as training.
    """
    Xc, unknown = _vec_cards_only(deckA, deckB)
    extras = _extra_feats(deckA, deckB)
    extra_names = app.state.extra_names or []
    if extra_names:
        Xe = np.array([[extras.get(name, 0.0) for name in extra_names]], dtype=np.float32)
        X = np.hstack([Xc, Xe])
    else:
        X = Xc
    return X, unknown, extras

@app.on_event("startup")
async def _startup():
    ok = _load_model_into_state()
    print(f"[startup] Model loaded: {ok} ({MODEL_PATH})")

# ----------------------------
# SQL
# ----------------------------
GET_DECK_ID_SQL = """
SELECT id
FROM decks
WHERE card_ids @> $1::int[] AND $1::int[] @> card_ids
LIMIT 1;
"""

TOP_DECKS_SQL = """
SELECT deck_id, card_ids, card_names, total_games, winrate
FROM deck_stats_named
WHERE total_games >= $1
ORDER BY winrate DESC
LIMIT $2;
"""

MOST_PLAYED_SQL = """
SELECT deck_id, card_ids, card_names, total_games, winrate
FROM deck_stats_named
ORDER BY total_games DESC
LIMIT $1;
"""

MATCHUP_SQL = """
SELECT a_deck_id, a_cards, b_deck_id, b_cards, games, a_winrate
FROM deck_matchups_named
WHERE a_deck_id = $1 AND b_deck_id = $2
LIMIT 1;
"""

NEIGHBORS_SQL = """
WITH cand AS (
  SELECT d.id AS deck_id,
         d.card_ids,
         (
           SELECT COUNT(*)
           FROM (SELECT unnest(d.card_ids)) AS x(val)
           INTERSECT
           SELECT unnest($1::int[])
         ) AS overlap
  FROM decks d
)
SELECT c.deck_id, c.card_ids, s.total_games, s.winrate, c.overlap
FROM cand c
JOIN deck_stats s ON s.deck_id = c.deck_id
WHERE c.overlap >= $2
ORDER BY c.overlap DESC, s.total_games DESC
LIMIT $3;
"""

# Admin: refresh materialized views (optionally concurrently)
async def refresh_views(concurrent: bool = False):
    clause = "CONCURRENTLY " if concurrent else ""
    sqls = [
        f"REFRESH MATERIALIZED VIEW {clause}deck_stats;",
        f"REFRESH MATERIALIZED VIEW {clause}deck_matchups;",
    ]
    pool = await get_pool()
    async with pool.acquire() as con:
        for q in sqls:
            await con.execute(q)

# ----------------------------
# Public endpoints
# ----------------------------
@app.get("/health")
async def health():
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.fetchval("SELECT 1;")
    return {
        "ok": True,
        "model_loaded": getattr(app.state, "model", None) is not None,
        "vocab_size": len(getattr(app.state, "vocab", []) or []),
        "extra_features": getattr(app.state, "extra_names", []),
        "neighbors": {
            "min_overlap": getattr(app.state, "neigh_min_ov", 7),
            "k": getattr(app.state, "neigh_k", 10),
        },
    }

@app.get("/decks/top")
async def top_decks(limit: int = 50, min_games: int = 50):
    pool = await get_pool()
    async with pool.acquire() as con:
        rows = await con.fetch(TOP_DECKS_SQL, min_games, limit)
    return [dict(r) for r in rows]

@app.get("/decks/most_played")
async def most_played(limit: int = 50):
    pool = await get_pool()
    async with pool.acquire() as con:
        rows = await con.fetch(MOST_PLAYED_SQL, limit)
    return [dict(r) for r in rows]

@app.post("/deck/resolve")
async def resolve_deck(card_ids: List[int]):
    cards = normalize_cards(card_ids)
    if len(cards) != 8:
        raise HTTPException(400, "Provide exactly 8 card IDs.")
    pool = await get_pool()
    async with pool.acquire() as con:
        deck_id = await con.fetchval(GET_DECK_ID_SQL, cards)
        if deck_id:
            return {"found": True, "deck_id": deck_id, "card_ids": cards}
        neigh = await con.fetch(NEIGHBORS_SQL, cards, 7, 10)
    return {"found": False, "card_ids": cards, "neighbors": [dict(r) for r in neigh]}

@app.get("/matchup/{a_deck}/{b_deck}")
async def matchup_by_ids(a_deck: int, b_deck: int):
    pool = await get_pool()
    async with pool.acquire() as con:
        row = await con.fetchrow(MATCHUP_SQL, a_deck, b_deck)
    if not row:
        raise HTTPException(404, "No matchup found for that pair.")
    return dict(row)

@app.post("/matchup")
async def matchup_by_cards(payload: Dict[str, Any]):
    deckA = normalize_cards(payload.get("deckA", []))
    deckB = normalize_cards(payload.get("deckB", []))
    if len(deckA) != 8 or len(deckB) != 8:
        raise HTTPException(400, "Both deckA and deckB must be 8 card IDs.")

    min_overlap = int(payload.get("min_overlap", 7))
    k_neigh = int(payload.get("neighbors", 10))

    pool = await get_pool()
    async with pool.acquire() as con:
        a_id = await con.fetchval(GET_DECK_ID_SQL, deckA)
        b_id = await con.fetchval(GET_DECK_ID_SQL, deckB)

        if a_id and b_id:
            row = await con.fetchrow(MATCHUP_SQL, a_id, b_id)
            if row:
                return {"method": "exact", **dict(row)}

        neighborsA = []
        neighborsB = []
        if not a_id:
            neighborsA = [dict(r) for r in await con.fetch(NEIGHBORS_SQL, deckA, min_overlap, k_neigh)]
        else:
            neighborsA = [{"deck_id": a_id, "card_ids": deckA, "total_games": None, "winrate": None, "overlap": 8}]
        if not b_id:
            neighborsB = [dict(r) for r in await con.fetch(NEIGHBORS_SQL, deckB, min_overlap, k_neigh)]
        else:
            neighborsB = [{"deck_id": b_id, "card_ids": deckB, "total_games": None, "winrate": None, "overlap": 8}]

        weights_sum = 0.0
        prob_sum = 0.0
        games_total = 0

        for na in neighborsA:
            for nb in neighborsB:
                row = await con.fetchrow(MATCHUP_SQL, na["deck_id"], nb["deck_id"])
                if not row:
                    continue
                w = float(na["overlap"]) * float(nb["overlap"])
                weights_sum += w
                prob_sum += w * float(row["a_winrate"])
                games_total += int(row["games"])

        if weights_sum > 0:
            return {
                "method": "neighbors",
                "estimate_a_winrate": prob_sum / weights_sum,
                "paired_matchup_games": games_total,
                "neighborsA": neighborsA,
                "neighborsB": neighborsB
            }

    return {"method": "fallback", "estimate_a_winrate": 0.5, "paired_matchup_games": 0}

# ----------------------------
# ML prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(payload: Dict[str, Any]):
    """
    ML prediction of P(A wins) using the trained bundle (cards + extra scalar features).
    Body:
      { "deckA": [8 ids], "deckB": [8 ids] }
    """
    if getattr(app.state, "model", None) is None:
        raise HTTPException(503, "ML model not loaded on server.")
    deckA = normalize_cards(payload.get("deckA", []))
    deckB = normalize_cards(payload.get("deckB", []))
    if len(deckA) != 8 or len(deckB) != 8:
        raise HTTPException(400, "Both deckA and deckB must be 8 card IDs.")

    X, unknown, extras = _vec_full(deckA, deckB)
    prob = float(app.state.model.predict_proba(X)[0, 1])  # P(A wins)

    return {
        "method": "ml",
        "prob_a_wins": prob,
        "prob_b_wins": 1.0 - prob,
        "unknown_cards": sorted(set(unknown)),
        "vocab_size": len(app.state.vocab),
        "features": {
            "extra_used": app.state.extra_names,
            **extras
        }
    }

# ----------------------------
# Admin endpoints
# ----------------------------
@app.post("/admin/refresh")
async def admin_refresh(
    concurrent: bool = False,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")
):
    """Refresh materialized views. Set concurrent=true if you created UNIQUE indexes."""
    require_admin(x_admin_token)
    try:
        await refresh_views(concurrent=concurrent)
    except asyncpg.PostgresError as e:
        raise HTTPException(500, f"Refresh failed: {e}")
    return {"ok": True, "concurrent": concurrent}

@app.post("/admin/reload_model")
async def admin_reload_model(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")
):
    """Reload deck_model.pkl without restarting the server."""
    require_admin(x_admin_token)
    ok = _load_model_into_state()
    if not ok:
        raise HTTPException(500, f"Failed to load model from {MODEL_PATH}")
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "vocab_size": len(app.state.vocab),
        "extra_features": app.state.extra_names,
    }
