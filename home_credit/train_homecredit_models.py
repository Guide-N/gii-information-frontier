# train_homecredit_models_full.py
"""
Train Models A/B/C/D using info_table_full.csv produced by gii_compute_homecredit_full.py
Saves models and homecredit_training_results/models_summary.csv
"""
import os, joblib, time, argparse
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

OUT_DIR = "homecredit_training_results"

def safe_label_encode_df(df, cat_cols):
    encs = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].fillna("__MISSING__").astype(str))
        encs[c] = le
    return encs

def compute_metrics(y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc = roc_auc_score(y_true, probs) if probs is not None else np.nan
    pr = average_precision_score(y_true, probs) if probs is not None else np.nan
    cm = confusion_matrix(y_true, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "pr_auc": pr, "confusion_matrix": cm.tolist()}

def train_and_eval(clf, Xtr, Xte, ytr, yte):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    t1 = time.time()
    try:
        probs = clf.predict_proba(Xte)[:,1]
    except:
        probs = None
    preds = clf.predict(Xte)
    metrics = compute_metrics(yte, preds, probs)
    metrics["train_time_sec"] = t1 - t0
    return metrics

def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading data:", args.input)
    df = pd.read_csv(args.input)
    if 'SK_ID_CURR' in df.columns:
        df = df.drop(columns=['SK_ID_CURR'])
    if 'TARGET' not in df.columns:
        raise SystemExit("TARGET not found.")
    y = df['TARGET'].astype(int)
    X = df.drop(columns=['TARGET']).copy()

    # load MI ranking
    info = pd.read_csv(args.mi, index_col=0)
    if 'mi_bits' not in info.columns:
        # find mi-like column
        col = [c for c in info.columns if 'mi' in c.lower()]
        if col:
            info.rename(columns={col[0]:'mi_bits'}, inplace=True)
        else:
            raise SystemExit("mi_bits column not found in MI file.")
    mi = info['mi_bits'].sort_values(ascending=False)

    K = args.top_k
    topK = mi.head(K).index.tolist()
    bottomK = mi.tail(K).index.tolist()
    half = K//2
    mixed = mi.head(half).index.tolist() + mi.tail(half).index.tolist()

    feature_sets = {
        "Model_A_all": X.columns.tolist(),
        "Model_B_topMI": topK,
        "Model_C_lowMI": bottomK,
        "Model_D_mixed": mixed
    }
    print("Feature sets sizes:", {k:len(v) for k,v in feature_sets.items()})

    # preprocess X
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    encs = safe_label_encode_df(X, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    results = []
    for name, feats in feature_sets.items():
        print("Training", name, "with", len(feats), "features")
        Xtr_sub = Xtr[feats].copy()
        Xte_sub = Xte[feats].copy()
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_sub)
        Xte_s = scaler.transform(Xte_sub)
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        metrics = train_and_eval(clf, Xtr_s, Xte_s, ytr, yte)
        joblib.dump(clf, os.path.join(OUT_DIR, f"{name}_logreg.joblib"))
        joblib.dump(scaler, os.path.join(OUT_DIR, f"{name}_scaler.joblib"))
        row = {"model": name, "n_features": len(feats)}
        row.update(metrics)
        results.append(row)
        print(f"{name} -> acc {metrics['accuracy']:.4f} f1 {metrics['f1']:.4f} roc {metrics['roc_auc']:.4f} pr {metrics['pr_auc']:.4f} train_time {metrics['train_time_sec']:.2f}s")

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "models_summary.csv"), index=False)
    print("Saved models_summary.csv to", OUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="application_train.csv")
    parser.add_argument("--mi", default=os.path.join("homecredit_results_full","info_table_full.csv"))
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
