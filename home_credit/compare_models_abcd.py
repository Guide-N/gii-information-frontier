# compare_models_abcd.py
"""
Compare LogisticRegression, RandomForest, LightGBM (if installed) on Models A/B/C/D.
- Uses stratified k-fold CV (default 5).
- Reads info_table_full.csv (MI) to pick top-K / bottom-K features.
- Saves numeric results to: homecredit_comparison/results_summary.csv
- Also saves pickled fitted models (optional - per fold / not by default).
"""
import os, time, joblib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# CONFIG
INPUT_CSV = "application_train.csv"
MI_CSV = os.path.join("homecredit_results_full","info_table_full.csv")
OUT_DIR = "homecredit_comparison"
N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 16

os.makedirs(OUT_DIR, exist_ok=True)

# load data
print("Loading data...")
df = pd.read_csv(INPUT_CSV)
if 'SK_ID_CURR' in df.columns:
    df = df.drop(columns=['SK_ID_CURR'])
if 'TARGET' not in df.columns:
    raise SystemExit("TARGET column not found.")
y = df['TARGET'].astype(int)
X = df.drop(columns=['TARGET']).copy()
print("Shape:", X.shape)

# preprocess: simple safe fills and encode categoricals for sklearn
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna("__MISSING__").astype(str)
    le = LabelEncoder(); X[c] = le.fit_transform(X[c])

# load MI ranking
info = pd.read_csv(MI_CSV, index_col=0)
mi_col = [c for c in info.columns if 'mi' in c.lower()]
if len(mi_col)==0:
    raise SystemExit("mi column not found in MI file.")
mi = info[mi_col[0]].sort_values(ascending=False)

# define feature sets
all_feats = X.columns.tolist()
topK = mi.head(TOP_K).index.tolist()
bottomK = mi.tail(TOP_K).index.tolist()
half = TOP_K // 2
mixed = mi.head(half).index.tolist() + mi.tail(half).index.tolist()

feature_sets = {
    "Model_A_all": all_feats,
    "Model_B_topMI": topK,
    "Model_C_lowMI": bottomK,
    "Model_D_mixed": mixed
}

print("Feature sets:", {k: len(v) for k,v in feature_sets.items()})

# models to evaluate
models = {}
models["LogReg"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1))
models["RandomForest"] = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)

# try LightGBM
try:
    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMClassifier(n_estimators=1000, n_jobs=-1, random_state=RANDOM_STATE)
    print("LightGBM detected and will be used.")
except Exception:
    print("LightGBM not installed; skipping LGBM. (install with pip install lightgbm)")

# metrics accumulator
rows = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fs_name, feats in feature_sets.items():
    Xsub = X[feats].values
    for mname, model in models.items():
        print(f"Running {mname} on {fs_name} | features={len(feats)}")
        # cross_val_predict to get out-of-fold probabilities
        # use method='predict_proba' where available
        try:
            probs = cross_val_predict(model, Xsub, y, cv=skf, method='predict_proba', n_jobs=-1)[:,1]
        except Exception:
            # fallback to manual fold loop
            probs = np.zeros(len(y))
            for fold_i, (tr, te) in enumerate(skf.split(Xsub, y)):
                m = joblib.clone(model)
                m.fit(Xsub[tr], y.iloc[tr])
                try:
                    probs[te] = m.predict_proba(Xsub[te])[:,1]
                except Exception:
                    probs[te] = m.predict(Xsub[te])  # degrade
        preds = (probs >= 0.5).astype(int)
        roc = roc_auc_score(y, probs)
        pr = average_precision_score(y, probs)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        rows.append({
            "model_family": mname,
            "feature_set": fs_name,
            "n_features": len(feats),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "accuracy": float(acc),
            "f1": float(f1)
        })
        # save an example fitted model on full train for inspection (optional)
        try:
            model.fit(Xsub, y)
            joblib.dump(model, os.path.join(OUT_DIR, f"{mname}_{fs_name}_fullfit.joblib"))
        except Exception:
            pass

# save results
res_df = pd.DataFrame(rows)
res_path = os.path.join(OUT_DIR, "results_summary.csv")
res_df.to_csv(res_path, index=False)
print("Saved results to", res_path)
