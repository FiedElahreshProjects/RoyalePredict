import os
import sys
import argparse
import asyncio
from typing import Optional

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_reports_dir(path: str = "reports") -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def fetch_df(sql: str, dsn: str) -> pd.DataFrame:
    with psycopg2.connect(dsn) as conn:
        return pd.read_sql(sql, conn)


def plot_top_decks(dsn: str, out_dir: str, limit: int = 15, min_games: int = 50) -> None:
    sql = f"""
        SELECT deck_id, card_names, total_games, winrate
        FROM deck_stats_named
        WHERE total_games >= {int(min_games)}
        ORDER BY winrate DESC
        LIMIT {int(limit)}
    """
    df = fetch_df(sql, dsn)
    if df.empty:
        print("[warn] deck_stats_named is empty or missing.")
        return
    df = df.copy()
    df["label"] = df.apply(lambda r: f"#{int(r.deck_id)}\n{', '.join(r.card_names[:2])}…", axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="winrate", y="label", color="#4C78A8")
    plt.xlabel("Winrate")
    plt.ylabel("Top decks (by winrate)")
    plt.xlim(0.45, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_decks_winrate.png"), dpi=150)
    plt.close()


def plot_most_played(dsn: str, out_dir: str, limit: int = 15) -> None:
    sql = f"""
        SELECT deck_id, card_names, total_games, winrate
        FROM deck_stats_named
        ORDER BY total_games DESC
        LIMIT {int(limit)}
    """
    df = fetch_df(sql, dsn)
    if df.empty:
        print("[warn] deck_stats_named is empty or missing.")
        return
    df = df.copy()
    df["label"] = df.apply(lambda r: f"#{int(r.deck_id)}\n{', '.join(r.card_names[:2])}…", axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="total_games", y="label", color="#72B7B2")
    plt.xlabel("Total games")
    plt.ylabel("Most played decks")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_decks_most_played.png"), dpi=150)
    plt.close()


def plot_winrate_distribution(dsn: str, out_dir: str, min_games: int = 25) -> None:
    sql = f"""
        SELECT winrate
        FROM deck_stats
        WHERE total_games >= {int(min_games)}
    """
    df = fetch_df(sql, dsn)
    if df.empty:
        print("[warn] deck_stats is empty or missing.")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df["winrate"], bins=30, kde=True, color="#E45756")
    plt.xlabel("Deck winrate")
    plt.ylabel("Count")
    plt.title("Distribution of deck winrates")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "winrate_distribution.png"), dpi=150)
    plt.close()


async def _compute_model_curves(out_dir: str, sample_limit: Optional[int] = None):
    # Optionally allow a smaller sample for quick runs by setting env BEFORE importing train_model
    if sample_limit is not None:
        os.environ["TRAIN_SAMPLE_LIMIT"] = str(int(sample_limit))

    import numpy as np
    from sklearn.metrics import roc_curve, RocCurveDisplay
    from sklearn.calibration import calibration_curve

    # Late import so env overrides apply
    import train_model as tm

    rows = await tm.fetch_rows()
    if not rows:
        print("[warn] No rows fetched from DB; cannot compute model curves.")
        return

    vocab, card_to_idx = tm.build_vocab(rows)
    X_sparse, y = tm.build_sparse_antisym(rows, card_to_idx)

    # Train model (fast grid if you reduce env grids)
    if tm.USE_XGB:
        model, metrics, split_idx = tm.fit_xgb_gpu_or_cpu(X_sparse, y, dense_stats=None)
    elif tm.USE_LR:
        model, metrics, split_idx = tm.fit_logreg_timeaware_sparse_antisym(X_sparse, y, d=len(vocab), dense_stats=None)
    else:
        model, metrics, split_idx = tm.fit_xgb_gpu_or_cpu(X_sparse, y, dense_stats=None)

    # Build test slice and predict
    Xte = X_sparse[split_idx:]
    yte = y[split_idx:]
    if Xte.shape[0] == 0:
        print("[warn] Empty test split; adjust TEST_FRACTION.")
        return

    proba_te = model.predict_proba(Xte)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(yte, proba_te)
    plt.figure(figsize=(6, 6))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC curve (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(yte, proba_te, n_bins=15, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="#888888", label="perfect")
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Calibration curve (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate charts and optional model curves.")
    parser.add_argument("--dsn", default=os.getenv("PG_DSN", "postgresql://postgres:Feeda123@localhost:5432/Clash"), help="Postgres DSN")
    parser.add_argument("--out", default="reports", help="Output directory for images")
    parser.add_argument("--limit", type=int, default=15, help="Top decks limit")
    parser.add_argument("--min-games", type=int, default=50, help="Min games filter for winrate charts")
    parser.add_argument("--with-model-curves", action="store_true", help="Also compute ROC & calibration by training a quick model")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional TRAIN_SAMPLE_LIMIT override for faster runs")
    args = parser.parse_args()

    out_dir = ensure_reports_dir(args.out)

    try:
        plot_top_decks(args.dsn, out_dir, limit=args.limit, min_games=args.min_games)
        plot_most_played(args.dsn, out_dir, limit=args.limit)
        plot_winrate_distribution(args.dsn, out_dir, min_games=max(25, args.min_games))
        print(f"Saved charts to: {out_dir}")
    except Exception as e:
        print(f"[error] Failed to generate DB charts: {e}", file=sys.stderr)

    if args.with_model_curves:
        try:
            asyncio.run(_compute_model_curves(out_dir, sample_limit=args.sample_limit))
            print(f"Saved model curves to: {out_dir}")
        except Exception as e:
            print(f"[warn] Skipped model curves: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


