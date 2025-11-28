import os
import time
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)

# -------------------------
# Config
# -------------------------
LOCAL_XLSX = "credit_default_dataset.xlsx"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
TARGET_CANDIDATES = [
    "default payment next month",
    "default.payment.next.month",
    "default_next_month",
    "default"
]
TOP_K = 8  # number of top features for Model B
LOW_K = 8  # number of bottom features for Model C
RANDOM_STATE = 42

# Which classifier to use?
CLASSIFIER = "logreg"  # options: "logreg" or "rf"

SAVE_MODELS = True
OUT_DIR = "model_results"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def load_dataset():
    if os.path.exists(LOCAL_XLSX):
        print(f"Loading local file: {LOCAL_XLSX}")
        try:
            df = pd.read_excel(LOCAL_XLSX, header=0)
        except Exception:
            df = pd.read_excel(LOCAL_XLSX, header=1)
    else:
        print("Local file not found — loading from UCI URL (requires internet).")
        df = pd.read_excel(UCI_URL, header=1)
    return df


def choose_target_column(df):
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            return t
    return df.columns[-1]


def build_model(kind="logreg"):
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs")
    elif kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    else:
        raise ValueError("Unknown model kind")


def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    probs = None
    try:
        probs = clf.predict_proba(X_test)[:, 1]
    except Exception:
        probs = None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if probs is not None else float("nan")
    pr_auc = average_precision_score(y_test, probs) if probs is not None else float("nan")
    cm = confusion_matrix(y_test, preds)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


# -------------------------
# Main flow
# -------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print("Columns:", list(df.columns)[:10], " ... total", len(df.columns))

    target = choose_target_column(df)
    print("Target column chosen:", target)

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    df = df.dropna(subset=[target])
    y = df[target].astype(int)
    X = df.drop(columns=[target]).copy()

    # Fill NaNs once
    X_filled = X.fillna(-9999)

    # ---------------------------------------------------------
    # CORRECTED LOGIC: SPLIT FIRST, THEN COMPUTE MI
    # ---------------------------------------------------------
    print("Splitting data 80/20 (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("Computing mutual information (Train features -> Train target)...")
    # Note: We use X_train.values to avoid any index alignment issues during calculation
    mi = mutual_info_classif(
        X_train.values,
        y_train.values,
        discrete_features='auto',
        random_state=RANDOM_STATE
    )

    # Map scores back to column names
    mi_series = pd.Series(mi, index=X_filled.columns).sort_values(ascending=False)

    # Save the MI scores for inspection
    mi_df = mi_series.reset_index().rename(columns={'index': 'feature', 0: 'mi'})
    mi_df.to_csv(os.path.join(OUT_DIR, "feature_mi.csv"), index=False)

    print("Top features by MI (Calculated on Train):")
    print(mi_series.head(12))

    # Define feature sets based on the TRAIN set ranking
    all_features = X_filled.columns.tolist()
    top_features = mi_series.head(TOP_K).index.tolist()
    low_features = mi_series.tail(LOW_K).index.tolist()

    # Mixed: half-high half-low
    half_high = mi_series.head(max(1, TOP_K // 2)).index.tolist()
    half_low = mi_series.tail(max(1, LOW_K // 2)).index.tolist()
    mixed_features = list(half_high) + list(half_low)

    print("\nFeature sets sizes:")
    print(f"Model A (all): {len(all_features)} features")
    print(f"Model B (top {TOP_K}): {len(top_features)} features")
    print(f"Model C (low {LOW_K}): {len(low_features)} features")
    print(f"Model D (mixed {len(mixed_features)}): {mixed_features}")


    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    def train_and_eval(feature_list, name):
        print("\n" + "=" * 50)
        print(f"Training {name}: {len(feature_list)} features")

        # Select columns from the already split X_train / X_test
        Xtr = X_train[feature_list]
        Xte = X_test[feature_list]

        clf = build_model(CLASSIFIER)

        t0 = time.time()
        clf.fit(Xtr, y_train)
        train_time = time.time() - t0

        metrics = evaluate_model(clf, Xte, y_test)
        metrics["train_time_sec"] = train_time
        metrics["n_features"] = len(feature_list)

        print(f"{name} results:")
        print(f" Train time (s): {train_time:.3f}")
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f" PR-AUC: {metrics['pr_auc']:.4f}")

        if SAVE_MODELS:
            fname = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_{CLASSIFIER}.joblib")
            joblib.dump(clf, fname)

        return {
            "model": name,
            "n_features": metrics["n_features"],
            "train_time_sec": metrics["train_time_sec"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"]
        }


    # Run experiments
    results = []
    results.append(train_and_eval(all_features, "Model_A_all"))
    results.append(train_and_eval(top_features, "Model_B_highMI"))
    results.append(train_and_eval(low_features, "Model_C_lowMI"))
    results.append(train_and_eval(mixed_features, "Model_D_mixed"))

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(OUT_DIR, "models_summary.csv"), index=False)

    print("\n=== Summary table ===")
    print(summary_df[["model", "n_features", "train_time_sec", "accuracy", "roc_auc", "pr_auc"]].to_string(index=False))
    print("\nAll done. Check folder:", OUT_DIR)