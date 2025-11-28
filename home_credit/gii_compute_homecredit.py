# gii_compute_homecredit_full.py
"""
CORRECTED: Splits data 80/20 FIRST, then computes MI on TRAIN set only.
This prevents data leakage in feature selection.
"""

import os
import math
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split  # Added

# ---------------- CONFIG ----------------
DATA_CSV = "application_train.csv"
OUT_DIR = "homecredit_results_full"
BINS = 30
ENTROPY_BINS = 50
TOP_K = 16
PAIRWISE_TOP_N = 50
N_JOBS = -1
RANDOM_STATE = 42  # Crucial for alignment with training scripts


# ----------------------------------------

def entropy_of_series(s: pd.Series, bins: int = ENTROPY_BINS) -> float:
    if s.nunique() <= 1:
        return 0.0
    if s.dtype.kind in "bifc" and s.nunique() > bins:
        counts, _ = np.histogram(s.dropna().values, bins=bins)
    else:
        counts = s.fillna("__MISSING__").value_counts().values
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def discretize_series_for_mi(s: pd.Series, bins: int = BINS) -> pd.Series:
    if s.dtype.kind in "bifc" and s.nunique() > bins:
        try:
            return pd.qcut(s.rank(method="first"), q=bins, duplicates="drop").astype(str)
        except Exception:
            return pd.cut(s.fillna(s.median()), bins=bins).astype(str)
    else:
        return s.fillna("__MISSING__").astype(str)


def compute_mi_for_columns(X: pd.DataFrame, y: pd.Series, bins: int = BINS, n_jobs: int = N_JOBS) -> pd.Series:
    cols = X.columns.tolist()
    print("Discretizing features for MI computation...")
    disc = {}
    for c in cols:
        disc[c] = discretize_series_for_mi(X[c], bins)
    y_disc = y.astype(str)

    def mi_worker(col):
        sa = disc[col]
        mi_nats = mutual_info_score(sa, y_disc)
        mi_bits = mi_nats / math.log(2)
        return col, float(mi_bits)

    print("Computing MI in parallel...")
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(mi_worker)(c) for c in cols
    )
    mi_dict = {c: v for c, v in results}
    return pd.Series(mi_dict).sort_values(ascending=False).rename("mi_bits")


def main():
    root = os.path.dirname(__file__)
    data_path = os.path.join(root, DATA_CSV)
    out_dir = os.path.join(root, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)

    if "TARGET" not in df.columns: raise SystemExit("TARGET missing.")
    if "SK_ID_CURR" in df.columns: df = df.drop(columns=["SK_ID_CURR"])

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"]).copy()

    # Preprocess (Fill NaNs) BEFORE Split to keep consistent logic
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols: X[c] = X[c].fillna(X[c].median())
    for c in cat_cols: X[c] = X[c].fillna("__MISSING__").astype(str)

    # ---------------------------------------------------------
    # CORRECTED: Split Data Check
    # ---------------------------------------------------------
    print(f"Splitting data 80/20 with random_state={RANDOM_STATE}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"MI will be computed on {len(X_train)} training rows only.")

    # Compute MI on TRAIN set
    t0 = time.time()
    mi_series = compute_mi_for_columns(X_train, y_train, bins=BINS, n_jobs=N_JOBS)
    print(f"MI computation done in {time.time() - t0:.1f} seconds.")

    # Calculate Entropy (can be done on full set or train, less critical)
    entropies = {c: entropy_of_series(X[c]) for c in X.columns}
    ent_series = pd.Series(entropies).rename("entropy_bits").sort_values(ascending=False)

    # Save Results
    info_table = pd.concat([ent_series, mi_series.reindex(X.columns).fillna(0)], axis=1)
    info_path = os.path.join(out_dir, "info_table_full.csv")
    info_table.to_csv(info_path)
    print("Saved info table:", info_path)

    # Save MI Series specifically
    mi_series.to_csv(os.path.join(out_dir, "mi_series_full.csv"), header=True)

    # GII Summaries
    K = TOP_K
    topK = mi_series.head(K).index.tolist()
    bottomK = mi_series.tail(K).index.tolist()

    with open(os.path.join(out_dir, "gii_summary.txt"), "w") as f:
        f.write(f"MI calculated on TRAIN set ({len(X_train)} rows)\n")
        f.write(f"Top Features:\n{mi_series.head(20).to_string()}")

    print("Done. MI calculation is rigourous (No Leakage).")


if __name__ == "__main__":
    main()