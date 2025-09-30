# train_model.py
import os
import time
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Iterable

from dotenv import load_dotenv
import asyncpg
import numpy as np
import joblib

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# ---------------- Env ----------------
load_dotenv()

# ---------------- Config ----------------
PG_DSN     = os.getenv("PG_DSN", "postgresql://postgres:Feeda123@localhost:5432/Clash")
MODEL_PATH = os.getenv("MODEL_PATH", "deck_model.pkl")

TEST_FRACTION  = float(os.getenv("TEST_FRACTION", "0.2"))   # last 20% = test (time-aware)
SAMPLE_LIMIT   = int(os.getenv("TRAIN_SAMPLE_LIMIT", "0"))  # 0 = all rows
RANDOM_SEED    = int(os.getenv("RANDOM_SEED", "42"))

# Trainer selection
USE_LR   = os.getenv("USE_LR", "0") == "1"
USE_XGB  = os.getenv("USE_XGB", "1") == "1" and not USE_LR

# Feature controls
USE_CROSS          = os.getenv("USE_CROSS", "1") == "1"        # include signed cross block
USE_HASHED_CROSS   = os.getenv("USE_HASHED_CROSS", "0") == "1" # hash cross features
CROSS_HASH_DIM     = int(os.getenv("CROSS_HASH_DIM", "4096"))  # width of hashed cross if enabled

# Dense deck stats (optional hand features)
USE_DENSE_STATS = os.getenv("USE_DENSE_STATS", "0") == "1"
CARD_META: Dict[int, Dict[str, float]] = {
    # Fill to enable dense stats:
    # 26000000: {"elixir": 4.0, "is_air": 0, "is_building": 0, "is_spell": 0},
}

# Logistic (baseline) hyperparams (only used when USE_LR=1)
def _parse_grid(env_name: str, default_vals):
    raw = os.getenv(env_name, "")
    if not raw.strip():
        return default_vals
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out or default_vals

LR_C_GRID   = _parse_grid("C_GRID",   [0.05, 0.1, 0.2, 0.5])
LR_L1R_GRID = _parse_grid("L1R_GRID", [0.5, 0.75])
LR_MAX_ITER = int(os.getenv("MAX_ITER", "800"))
LR_TOL      = float(os.getenv("TOL", "1e-3"))
LR_VERBOSE  = int(os.getenv("EPOCH_VERBOSE", "0"))

# ---------------- XGBoost hyperparams ----------------
def _parse_float_list(env, default):
    raw = os.getenv(env, "")
    if not raw.strip():
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]

def _parse_int_list(env, default):
    raw = os.getenv(env, "")
    if not raw.strip():
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

XGB_DEPTHS      = _parse_int_list  ("XGB_DEPTHS",    [4, 6])          # leaner by default
XGB_N_EST       = _parse_int_list  ("XGB_N_EST",     [800])
XGB_LR          = _parse_float_list("XGB_LR",        [0.05, 0.1])
XGB_L1          = _parse_float_list("XGB_L1",        [0.0, 0.5])
XGB_L2          = _parse_float_list("XGB_L2",        [1.0, 5.0, 10.0])
XGB_SUBSAMPLE   = _parse_float_list("XGB_SUBSAMPLE", [0.6, 0.8])
XGB_COLSAMPLE   = _parse_float_list("XGB_COLSAMPLE", [0.6, 0.8])
XGB_MIN_CHILD   = _parse_float_list("XGB_MIN_CHILD", [1.0, 5.0, 10.0])
XGB_GAMMA       = _parse_float_list("XGB_GAMMA",     [0.0, 1.0])
XGB_EARLY_STOP  = int(os.getenv("XGB_EARLY_STOP", "50"))
XGB_VERBOSITY   = int(os.getenv("XGB_VERBOSITY", "0"))  # 0 silent
# Metric used for early stopping/selection within xgboost. We'll still report AUC.
XGB_SELECT_METRIC = os.getenv("XGB_SELECT_METRIC", "logloss")  # "logloss" or "auc"

# ---------------- SQL ----------------
SQL = """
SELECT
  m.ts,
  CASE WHEN m.a_won THEN 1 ELSE 0 END AS y,
  da.card_ids AS a_cards,
  db.card_ids AS b_cards
FROM matches m
JOIN decks da ON da.id = m.a_deck_id
JOIN decks db ON db.id = m.b_deck_id
ORDER BY m.ts ASC
"""

@dataclass
class Row:
    ts: Any
    y: int
    a_cards: List[int]
    b_cards: List[int]

# ---------------- Data IO ----------------
async def fetch_rows() -> List[Row]:
    q = SQL + (f"\nLIMIT {SAMPLE_LIMIT}" if SAMPLE_LIMIT > 0 else "")
    t0 = time.perf_counter()
    print(f"[DB] Connecting")
    pool = await asyncpg.create_pool(dsn=PG_DSN, min_size=1, max_size=5)
    try:
        async with pool.acquire() as con:
            print(f"[DB] Running query (limit={SAMPLE_LIMIT})…")
            recs = await con.fetch(q)
    finally:
        await pool.close()
    dt = time.perf_counter() - t0
    print(f"[DB] Done. Fetched {len(recs):,} rows in {dt:.2f}s.")
    return [Row(ts=r["ts"], y=int(r["y"]), a_cards=list(r["a_cards"]), b_cards=list(r["b_cards"])) for r in recs]

# ---------------- Vocab ----------------
DeckKey = Tuple[int, ...]  # sorted 8-card tuple

def deck_key(cards: Iterable[int]) -> DeckKey:
    return tuple(sorted(int(c) for c in cards))

def build_vocab(rows: List[Row]) -> Tuple[List[int], Dict[int, int]]:
    vocab = sorted({cid for r in rows for cid in r.a_cards} |
                   {cid for r in rows for cid in r.b_cards})
    card_to_idx = {cid: i for i, cid in enumerate(vocab)}
    return vocab, card_to_idx

# ---------------- Antisymmetric Features ----------------
def _hash_pair(i: int, j: int, mod: int) -> int:
    # Simple, decent 2D hash; mod is CROSS_HASH_DIM
    return (i * 257 + j * 911) % mod

def build_sparse_antisym(
    rows: List[Row],
    card_to_idx: Dict[int, int],
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Antisymmetric features:
      - diff onehot: x_diff = onehot(A) - onehot(B) (shape d)
      - signed cross: for each (i in A, j in B): +1 at (i,j) and -1 at (j,i)
        * either full d*d or hashed to CROSS_HASH_DIM
    """
    t0 = time.perf_counter()
    n = len(rows)
    d = len(card_to_idx)
    print(f"[FEAT] Building antisymmetric features for {n:,} rows, d={d}…")

    # COO builders
    r_diff, c_diff, v_diff = [], [], []
    r_cross, c_cross, v_cross = [], [], []

    y = np.zeros(n, dtype=np.int32)
    ping_every = 25000 if n > 25000 else 0

    cross_dim = 0
    if USE_CROSS:
        cross_dim = CROSS_HASH_DIM if USE_HASHED_CROSS else d * d

    for r_i, r in enumerate(rows):
        if ping_every and (r_i % ping_every == 0) and r_i > 0:
            print(f"[FEAT] … {r_i:,}/{n:,} rows")

        y[r_i] = r.y
        A = {card_to_idx.get(int(c)) for c in set(r.a_cards) if int(c) in card_to_idx}
        B = {card_to_idx.get(int(c)) for c in set(r.b_cards) if int(c) in card_to_idx}
        A.discard(None); B.discard(None)

        # diff onehot = A - B
        for i in A:
            r_diff.append(r_i); c_diff.append(i); v_diff.append(+1.0)
        for j in B:
            r_diff.append(r_i); c_diff.append(j); v_diff.append(-1.0)

        if USE_CROSS:
            if USE_HASHED_CROSS:
                for i in A:
                    for j in B:
                        r_cross.append(r_i); c_cross.append(_hash_pair(i, j, CROSS_HASH_DIM)); v_cross.append(+1.0)
                        r_cross.append(r_i); c_cross.append(_hash_pair(j, i, CROSS_HASH_DIM)); v_cross.append(-1.0)
            else:
                for i in A:
                    base_i = i * d
                    for j in B:
                        base_j = j * d
                        r_cross.append(r_i); c_cross.append(base_i + j); v_cross.append(+1.0)
                        r_cross.append(r_i); c_cross.append(base_j + i); v_cross.append(-1.0)

    X_diff = csr_matrix((v_diff, (r_diff, c_diff)), shape=(n, d), dtype=np.float32)

    if USE_CROSS:
        X_cross = csr_matrix((v_cross, (r_cross, c_cross)),
                             shape=(n, cross_dim), dtype=np.float32)
        X = hstack([X_diff, X_cross], format="csr")
    else:
        X = X_diff

    dt = time.perf_counter() - t0
    print(f"[FEAT] Done in {dt:.2f}s. X shape={X.shape}, nnz={X.nnz:,}")
    return X, y

# ---------------- Dense stats (optional) ----------------
def deck_stats(cards: Iterable[int], meta: Dict[int, Dict[str, float]]) -> np.ndarray:
    cards = list(set(int(c) for c in cards))
    if not cards:
        return np.zeros(6, dtype=np.float32)
    elixirs = [meta.get(c, {}).get("elixir", 3.5) for c in cards]
    air = sum(meta.get(c, {}).get("is_air", 0) for c in cards)
    bld = sum(meta.get(c, {}).get("is_building", 0) for c in cards)
    spl = sum(meta.get(c, {}).get("is_spell", 0) for c in cards)
    return np.array([
        float(np.mean(elixirs)), float(np.std(elixirs)),
        float(air), float(bld), float(spl),
        float(len(cards))
    ], dtype=np.float32)

def build_dense_stats(rows: List[Row], meta: Dict[int, Dict[str, float]]) -> np.ndarray:
    t0 = time.perf_counter()
    feats = []
    for r in rows:
        sa = deck_stats(r.a_cards, meta)
        sb = deck_stats(r.b_cards, meta)
        feats.append(sa - sb)
    Xd = np.vstack(feats) if feats else np.zeros((0, 6), dtype=np.float32)
    dt = time.perf_counter() - t0
    print(f"[FEAT] Dense stats built in {dt:.2f}s. Shape={Xd.shape}")
    return Xd

# ---------------- Helpers ----------------
def _choose_train_val_split_len(y_train: np.ndarray) -> int:
    ntr = len(y_train)
    if ntr <= 2:
        return max(ntr - 1, 1)
    v_idx = int(0.9 * ntr)
    if v_idx <= 0 or v_idx >= ntr:
        v_idx = max(ntr - max(1, ntr // 5), 1)  # ~80/20 fallback
    return v_idx

# ---------------- Logistic Baseline (optional) ----------------
def fit_logreg_timeaware_sparse_antisym(
    X: csr_matrix, y: np.ndarray, d: int, dense_stats: np.ndarray = None
):
    from sklearn.linear_model import LogisticRegression

    print(f"[SPLIT] TEST_FRACTION={TEST_FRACTION}")
    n = len(y)
    split_idx = int((1.0 - TEST_FRACTION) * n)
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]
    print(f"[SPLIT] Train={len(ytr):,}, Test={len(yte):,}")

    if dense_stats is not None and dense_stats.shape[0] == n:
        from scipy.sparse import csr_matrix as _csr
        print("[FEAT] Appending dense stats to sparse matrix…")
        Xdense = _csr(dense_stats, dtype=np.float32)
        Xtr = hstack([Xtr, Xdense[:split_idx]], format="csr")
        Xte = hstack([Xte, Xdense[split_idx:]], format="csr")
        print(f"[FEAT] New shapes → Xtr={Xtr.shape}, Xte={Xte.shape}")

    v_idx = _choose_train_val_split_len(ytr)
    Xfit, Xval = Xtr[:v_idx], Xtr[v_idx:]
    yfit, yval = ytr[:v_idx], ytr[v_idx:]
    print(f"[SPLIT] Fit={len(yfit):,}, Val={len(yval):,}")

    # Simple grid (you can expand via env)
    best_model, best_auc = None, -1.0
    for l1r in LR_L1R_GRID:
        warm_model = None
        for C in sorted(LR_C_GRID):
            clf = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=l1r,
                C=C,
                solver="saga",
                max_iter=LR_MAX_ITER,
                tol=LR_TOL,
                class_weight="balanced",
                fit_intercept=True,
                warm_start=True,
                random_state=RANDOM_SEED,
                verbose=LR_VERBOSE,
            )
            if warm_model is not None:
                clf.coef_ = warm_model.coef_.copy()
                clf.intercept_ = warm_model.intercept_.copy()
                clf.classes_ = warm_model.classes_
            clf.fit(Xfit, yfit)
            warm_model = clf
            val_auc = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
            print(f"[LR] C={C}, l1r={l1r} → Val AUC={val_auc:.4f}")
            if val_auc > best_auc:
                best_model, best_auc = clf, val_auc

    print("[CAL] Calibrating probabilities (sigmoid, prefit)…")
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
    calibrated.fit(Xval, yval)
    print("[CAL] Done.")

    # Evaluate (guard test emptiness)
    proba_tr = calibrated.predict_proba(Xtr)[:, 1]
    if Xte.shape[0] > 0:
        proba_te = calibrated.predict_proba(Xte)[:, 1]
        pred_te  = (proba_te >= 0.5).astype(int)
        test_logloss = float(log_loss(yte, proba_te)) if len(np.unique(yte)) > 1 else float('nan')
        test_auc     = float(roc_auc_score(yte, proba_te)) if len(np.unique(yte)) > 1 else float('nan')
        test_acc     = float(accuracy_score(yte, pred_te))
    else:
        test_logloss = test_auc = test_acc = float('nan')

    metrics = {
        "backend": "logreg_saga",
        "val_auc": float(best_auc),
        "test_logloss": test_logloss,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "train_logloss": float(log_loss(ytr, proba_tr)) if len(np.unique(ytr)) > 1 else float('nan'),
        "train_auc": float(roc_auc_score(ytr, proba_tr)) if len(np.unique(ytr)) > 1 else float('nan'),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    print("[EVAL] Logistic baseline evaluated.")
    split_idx = int((1.0 - TEST_FRACTION) * len(y))
    return calibrated, metrics, split_idx

# ---------------- XGBoost (recommended) ----------------
class _PrefitXGBClassifier(BaseEstimator, ClassifierMixin):
    """A tiny sklearn-compatible wrapper around a prefit xgboost Booster."""
    def __init__(self, booster, best_iter):
        self.booster = booster
        self.best_iter = int(best_iter)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        # prefit; nothing to do
        return self

    def predict_proba(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        p = self.booster.predict(d, iteration_range=(0, self.best_iter + 1))
        return np.vstack([1.0 - p, p]).T

# ---- Top-level helpers for Platt calibrator so it's picklable ----
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

class _PlattCalibrated(BaseEstimator, ClassifierMixin):
    """Top-level, picklable wrapper that applies Platt scaling to a prefit XGB booster."""
    def __init__(self, base: _PrefitXGBClassifier, platt):
        self.base = base
        self.platt = platt
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self._estimator_type = "classifier"

    def fit(self, X, y=None):  # prefit wrapper
        return self

    def predict_proba(self, X):
        import xgboost as xgb
        raw = self.base.booster.predict(
            xgb.DMatrix(X),
            iteration_range=(0, self.base.best_iter + 1),
        )
        feat = _logit(raw).reshape(-1, 1)
        p1 = self.platt.predict_proba(feat)[:, 1]
        return np.vstack([1 - p1, p1]).T

def fit_xgb_gpu_or_cpu(X: csr_matrix, y: np.ndarray, dense_stats: np.ndarray = None):
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    print(f"[SPLIT] TEST_FRACTION={TEST_FRACTION}")
    n = len(y)
    split_idx = int((1.0 - TEST_FRACTION) * n)
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]
    print(f"[SPLIT] Train={len(ytr):,}, Test={len(yte):,}")

    # Optional dense stats
    if dense_stats is not None and dense_stats.shape[0] == n and dense_stats.shape[1] > 0:
        from scipy.sparse import csr_matrix as _csr
        print("[FEAT] Appending dense stats to sparse matrix…")
        Xdense = _csr(dense_stats, dtype=np.float32)
        Xtr = hstack([Xtr, Xdense[:split_idx]], format="csr")
        Xte = hstack([Xte, Xdense[split_idx:]], format="csr")
        print(f"[FEAT] New shapes → Xtr={Xtr.shape}, Xte={Xte.shape}")

    # Train/Val split inside TRAIN (last 10% for validation)
    v_idx = _choose_train_val_split_len(ytr)
    Xfit, Xval = Xtr[:v_idx], Xtr[v_idx:]
    yfit, yval = ytr[:v_idx], ytr[v_idx:]
    print(f"[SPLIT] Fit={len(yfit):,}, Val={len(yval):,}")

    dtrain = xgb.DMatrix(Xfit, label=yfit)
    dval   = xgb.DMatrix(Xval, label=yval)

    device = os.getenv("XGB_DEVICE", "").strip().lower() or "cuda"  # set XGB_DEVICE=cpu to force CPU
    if device not in {"cuda", "cpu"}:
        device = "cuda"
    print(f"[XGB] device={device} (set XGB_DEVICE=cpu to force CPU)")
    tree_method = "hist"  # with device=cuda, this is GPU-accelerated on XGB 2.x

    # Build grid
    grids = []
    for depth in XGB_DEPTHS:
        for lr in XGB_LR:
            for a in XGB_L1:
                for l in XGB_L2:
                    for ss in XGB_SUBSAMPLE:
                        for cs in XGB_COLSAMPLE:
                            for mcw in XGB_MIN_CHILD:
                                for gm in XGB_GAMMA:
                                    for n_est in XGB_N_EST:
                                        grids.append(dict(
                                            max_depth=depth, eta=lr, reg_alpha=a, reg_lambda=l,
                                            subsample=ss, colsample_bytree=cs, num_boost_round=n_est,
                                            min_child_weight=mcw, gamma=gm
                                        ))

    best_booster = None
    best_metrics = None
    best_params  = None
    best_iter    = None
    fit_count = 0

    for g in grids:
        fit_count += 1
        params = {
            "objective": "binary:logistic",
            "eval_metric": XGB_SELECT_METRIC,  # selection metric
            "tree_method": tree_method,
            "device": device,
            "max_depth": g["max_depth"],
            "eta": g["eta"],
            "subsample": g["subsample"],
            "colsample_bytree": g["colsample_bytree"],
            "reg_alpha": g["reg_alpha"],
            "reg_lambda": g["reg_lambda"],
            "min_child_weight": g["min_child_weight"],
            "gamma": g["gamma"],
            "seed": RANDOM_SEED,
        }
        num_boost_round = g["num_boost_round"]

        print(f"\n=== XGB FIT {fit_count} === params={params} num_boost_round={num_boost_round}")
        t_fit = time.perf_counter()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "validation")],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose_eval=False,
        )
        dt_fit = time.perf_counter() - t_fit

        # Best iteration index (0-based); use iteration_range for prediction in 2.x
        this_best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
        val_pred = booster.predict(dval, iteration_range=(0, this_best_iter + 1))
        val_auc = roc_auc_score(yval, val_pred)
        print(f"[XGB] Val AUC={val_auc:.4f} in {dt_fit/60:.2f} min (best_iter={this_best_iter})")

        if (best_metrics is None) or (val_auc > best_metrics["val_auc"]):
            best_booster = booster
            best_metrics = {"val_auc": float(val_auc)}
            best_params  = {**params, "num_boost_round": num_boost_round}
            best_iter    = this_best_iter

    # Wrap best booster as a sklearn classifier (prefit)
    base = _PrefitXGBClassifier(best_booster, best_iter)

    # --- Manual Platt scaling on the held-out validation slice ---
    print("[CAL] Calibrating probabilities via Platt scaling (logistic on validation)…")
    val_scores = best_booster.predict(xgb.DMatrix(Xval), iteration_range=(0, best_iter + 1))
    val_feat = _logit(val_scores).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(val_feat, yval)
    print("[CAL] Done.")

    calibrated = _PlattCalibrated(base, platt)

    # Evaluate calibrated model
    print("[EVAL] Evaluating on train/test…")
    proba_tr = calibrated.predict_proba(Xtr)[:, 1]

    if Xte.shape[0] > 0:
        proba_te = calibrated.predict_proba(Xte)[:, 1]
        pred_te  = (proba_te >= 0.5).astype(int)
        test_logloss = float(log_loss(yte, proba_te)) if len(np.unique(yte)) > 1 else float('nan')
        test_auc     = float(roc_auc_score(yte, proba_te)) if len(np.unique(yte)) > 1 else float('nan')
        test_acc     = float(accuracy_score(yte, pred_te))
    else:
        test_logloss = test_auc = test_acc = float('nan')

    metrics = {
        "backend": f"xgboost_{device}",
        "val_auc": float(best_metrics["val_auc"]),
        "test_logloss": test_logloss,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "train_logloss": float(log_loss(ytr, proba_tr)) if len(np.unique(ytr)) > 1 else float('nan'),
        "train_auc": float(roc_auc_score(ytr, proba_tr)) if len(np.unique(ytr)) > 1 else float('nan'),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "best_params": {**best_params, "best_iteration": int(best_iter)},
    }

    dt_total = time.perf_counter() - t0
    print(f"[EVAL] Done. Total train time: {dt_total/60:.2f} min")
    split_idx = int((1.0 - TEST_FRACTION) * len(y))
    return calibrated, metrics, split_idx

# ---------------- Main ----------------
async def main():
    print(f"[ENV] TRAIN_SAMPLE_LIMIT={SAMPLE_LIMIT} | TEST_FRACTION={TEST_FRACTION}")
    print(f"[ENV] USE_XGB={USE_XGB} | USE_LR={USE_LR} | RANDOM_SEED={RANDOM_SEED}")
    print(f"[ENV] USE_CROSS={USE_CROSS} | USE_HASHED_CROSS={USE_HASHED_CROSS} | CROSS_HASH_DIM={CROSS_HASH_DIM}")
    rows = await fetch_rows()
    print(f"Fetched {len(rows)} matches from DB.")
    if not rows:
        print("No rows found.")
        return

    vocab, card_to_idx = build_vocab(rows)
    d = len(vocab)
    print(f"Card vocabulary size: {d}")

    print("Building features (antisymmetric)…")
    X_sparse, y = build_sparse_antisym(rows, card_to_idx)

    dense_stats = None
    if USE_DENSE_STATS:
        dense_stats = build_dense_stats(rows, CARD_META)

    if USE_XGB:
        print("[TRAIN] Using XGBoost (GPU if available) for best accuracy.")
        model, metrics, split_idx = fit_xgb_gpu_or_cpu(X_sparse, y, dense_stats=dense_stats)
    elif USE_LR:
        print("[TRAIN] Using logistic regression (baseline).")
        model, metrics, split_idx = fit_logreg_timeaware_sparse_antisym(X_sparse, y, d, dense_stats=dense_stats)
    else:
        print("[TRAIN] Defaulting to XGBoost (set USE_LR=1 to force logistic).")
        model, metrics, split_idx = fit_xgb_gpu_or_cpu(X_sparse, y, dense_stats=dense_stats)

    print("Metrics:", {"split_index": split_idx, **metrics})

    total_dim = X_sparse.shape[1] + (0 if dense_stats is None else dense_stats.shape[1])

    bundle = {
        "model": model,         # Calibrated model (Platt over XGB or Calibrated LR)
        "vocab": vocab,
        "featurizer": "antisymmetric_diff+signed_cross_v1" if USE_CROSS and not USE_HASHED_CROSS
                      else ("antisymmetric_diff+hashed_signed_cross_v1" if USE_CROSS and USE_HASHED_CROSS
                            else "antisymmetric_diff_only_v1"),
        "details": {
            "dims": {"cards": d, "features_total": int(total_dim)},
            "split": {"test_fraction": TEST_FRACTION, "split_index": int(split_idx)},
            "trainer": metrics.get("backend", "unknown"),
            "trainer_params": metrics.get("best_params", {}),
            "calibrated": True,
            "calibration": "platt_logistic_on_val" if USE_XGB else "sklearn_sigmoid_prefit",
            "random_seed": RANDOM_SEED,
        }
    }
    print(f"[SAVE] Saving model to {MODEL_PATH} …")
    joblib.dump(bundle, MODEL_PATH)
    print(f"[SAVE] Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
