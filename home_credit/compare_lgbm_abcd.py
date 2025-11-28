# compare_lgbm_abcd_fixed.py
"""
Evaluate LightGBM on Models A/B/C/D using 5-fold stratified CV.
Fixed: avoid invalid 'verbose' kw for LGBMClassifier.fit()
Saves: homecredit_comparison/lgbm_results_summary.csv and full-fit models.
"""
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Config
INPUT_CSV = "application_train.csv"
MI_CSV = os.path.join("homecredit_results_full", "info_table_full.csv")
OUT_DIR = "homecredit_comparison"
N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 16
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

os.makedirs(OUT_DIR, exist_ok=True)

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])
    if "TARGET" not in df.columns:
        raise SystemExit("TARGET column not found in CSV.")
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"]).copy()
    # Separate numeric and categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Detected {len(num_cols)} numeric cols, {len(cat_cols)} categorical cols.")
    # Impute numeric medians
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    # Fill categorical and label-encode
    encoders = {}
    for c in cat_cols:
        X[c] = X[c].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le
    return X, y, num_cols, cat_cols, encoders

def load_mi_ranking():
    info = pd.read_csv(MI_CSV, index_col=0)
    mi_col = [c for c in info.columns if "mi" in c.lower()]
    if len(mi_col) == 0:
        raise SystemExit("mi_bits column not found in MI CSV.")
    mi_series = info[mi_col[0]].fillna(0).sort_values(ascending=False)
    return mi_series

def evaluate_lgbm_on_sets(X, y, mi_series):
    all_feats = X.columns.tolist()
    topK = mi_series.head(TOP_K).index.tolist()
    bottomK = mi_series.tail(TOP_K).index.tolist()
    half = TOP_K // 2
    mixed = mi_series.head(half).index.tolist() + mi_series.tail(half).index.tolist()

    feature_sets = {
        "Model_A_all": all_feats,
        "Model_B_topMI": topK,
        "Model_C_lowMI": bottomK,
        "Model_D_mixed": mixed
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, feats in feature_sets.items():
        print(f"\nEvaluating LightGBM on {name} ({len(feats)} features)...")
        Xsub = X[feats].values  # LightGBM accepts numpy arrays
        # Out-of-fold probabilities (predict_proba)
        print("  Running cross-validated predictions (oof)...")
        try:
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            oof_probs = cross_val_predict(clf, Xsub, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        except Exception as e:
            print("  cross_val_predict failed, falling back to manual CV. Error:", e)
            oof_probs = np.zeros(len(y))
            for fold, (tr, te) in enumerate(skf.split(Xsub, y)):
                clf = lgb.LGBMClassifier(**LGB_PARAMS)
                # Use early stopping on validation fold to avoid overfitting if wanted
                clf.fit(Xsub[tr], y.iloc[tr],
                        eval_set=[(Xsub[te], y.iloc[te])],
                        eval_metric="auc",
                        early_stopping_rounds=50)
                oof_probs[te] = clf.predict_proba(Xsub[te])[:, 1]

        preds = (oof_probs >= 0.5).astype(int)
        roc = roc_auc_score(y, oof_probs)
        pr = average_precision_score(y, oof_probs)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)

        # Save a full-fit model trained on the entire dataset (for inspection)
        print("  Training full-fit LightGBM on entire set (for saving)...")
        full_clf = lgb.LGBMClassifier(**LGB_PARAMS)
        # Do not pass verbose kw; use default fit API
        full_clf.fit(Xsub, y)
        model_path = os.path.join(OUT_DIR, f"lgbm_{name}_fullfit.joblib")
        joblib.dump(full_clf, model_path)

        row = {
            "model_family": "LightGBM",
            "feature_set": name,
            "n_features": len(feats),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "accuracy": float(acc),
            "f1": float(f1),
            "model_path": model_path
        }
        results.append(row)
        print(f"  {name} -> ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}  acc: {acc:.4f}  f1: {f1:.4f}")

    res_df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "lgbm_results_summary.csv")
    res_df.to_csv(out_path, index=False)
    print("\nSaved LightGBM results to:", out_path)
    return res_df

def main():
    X, y, num_cols, cat_cols, encoders = load_and_preprocess()
    mi_series = load_mi_ranking()
    res = evaluate_lgbm_on_sets(X, y, mi_series)
    print("\nAll done.")

if __name__ == "__main__":
    main()
