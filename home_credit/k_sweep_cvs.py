# k_sweep_cv.py
import os, time, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

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
for c in cat_cols:
    le = LabelEncoder(); X[c] = le.fit_transform(X[c])

mi = pd.read_csv(MI_CSV, index_col=0)['mi_bits'].sort_values(ascending=False)
K_list = [8,16,32,64]
results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for K in K_list:
    feats = mi.head(K).index.tolist()
    Xsub = X[feats].values
    # Standardize for LR
    scaler = StandardScaler().fit(Xsub); Xsub_s = scaler.transform(Xsub)
    # Logistic Regression CV
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    probs_lr = cross_val_predict(lr, Xsub_s, y, cv=skf, method='predict_proba')[:,1]
    roc_lr = roc_auc_score(y, probs_lr); pr_lr = average_precision_score(y, probs_lr)
    # RandomForest CV (no scaling needed)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    probs_rf = cross_val_predict(rf, Xsub, y, cv=skf, method='predict_proba')[:,1]
    roc_rf = roc_auc_score(y, probs_rf); pr_rf = average_precision_score(y, probs_rf)
    print(f"K={K} | LR ROC={roc_lr:.4f} PR={pr_lr:.4f} | RF ROC={roc_rf:.4f} PR={pr_rf:.4f}")
    results.append({"K":K,"lr_roc":roc_lr,"lr_pr":pr_lr,"rf_roc":roc_rf,"rf_pr":pr_rf})
pd.DataFrame(results).to_csv(os.path.join(OUT,"k_sweep_cv_results.csv"), index=False)
print("Saved k_sweep_cv_results.csv")
