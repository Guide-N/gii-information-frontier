# model_e_add_low.py
import os, joblib, time, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

INPUT = "application_train.csv"
MI_CSV = os.path.join("homecredit_results_full","info_table_full.csv")
OUT = "homecredit_training_results"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT)
if 'SK_ID_CURR' in df.columns: df = df.drop(columns=['SK_ID_CURR'])
y = df['TARGET'].astype(int)
X = df.drop(columns=['TARGET']).copy()

# preprocess
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for c in num_cols: X[c] = X[c].fillna(X[c].median())
for c in cat_cols: X[c] = X[c].fillna("__MISSING__").astype(str)
# encode categoricals
encs = {}
from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    le = LabelEncoder(); X[c]=le.fit_transform(X[c]); encs[c]=le

mi = pd.read_csv(MI_CSV, index_col=0)['mi_bits'].sort_values(ascending=False)
K = 16
topK = mi.head(K).index.tolist()
bottomK = mi.tail(K).index.tolist()

results = []
for extra in bottomK:
    feats = topK + [extra]
    Xs = X[feats]
    # train/test split (same as before)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    t0 = time.time(); clf.fit(Xtr_s, ytr); t = time.time()-t0
    try: probs = clf.predict_proba(Xte_s)[:,1]
    except: probs = None
    preds = clf.predict(Xte_s)
    row = {
        "added_feature": extra,
        "accuracy": accuracy_score(yte,preds),
        "f1": f1_score(yte,preds, zero_division=0),
        "roc_auc": roc_auc_score(yte, probs) if probs is not None else np.nan,
        "pr_auc": average_precision_score(yte, probs) if probs is not None else np.nan,
        "train_time_sec": t
    }
    print(row)
    results.append(row)

pd.DataFrame(results).to_csv(os.path.join(OUT, "model_e_add_low_results.csv"), index=False)
print("Saved model_e_add_low_results.csv")
