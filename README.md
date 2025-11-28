# THE GLOBAL INFORMATION INDEX: THE ILLUSION OF MODEL CAPACITY

Official code repository for the preprint: **"The Global Information Index: The Illusion of Model Capacity"** by Guide A. Ndapasowa and Jabulani Chibaya (GourdAI, 2025).

The Global Information Index (GII) quantifies the available predictive signal (the **Information Frontier**) in a dataset *before* model training.

---

## 🔗 Publication and Citation

* **Published Paper Link (DOI):** [INSERT ZENODO/SSRN LINK HERE]
* **Authors:** Guide A. Ndapasowa and Jabulani Chibaya (GourdAI)

## 🛠 Methodology and Rigor Audit

This repository implements the GII framework on the UCI Credit Default and Home Credit Default Risk datasets.

* **Audit Status:** The code is strictly audited to ensure the **Mutual Information (MI) feature ranking is calculated EXCLUSIVELY on the training subset**, eliminating data leakage (Peeking) in feature selection.
* **Core Finding:** Low-MI features consistently yield near-random performance (ROC-AUC approximately 0.53), proving that signal cannot be recovered where (GII) predicts none exists.
* **GII Metric:** Feature ranking uses the **KSG Mutual Information Estimator** (k=3).
* **Preprocessing:** Categorical features are handled via **integer encoding**.

## 🚀 How to Run the Experiments

### Setup

1.  Clone this repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  **Data:** Place `application_train.csv` (Home Credit) in the `/home_credit/` folder. The UCI data is downloaded automatically.

### Execution

1.  **Home Credit (The Big Test):** Run the MI calculation first, then the comparison.
    ```bash
    cd home_credit
    python gii_compute_homecredit.py  # Calculates MI on 80% train set
    python compare_lgbm_abcd.py       # Runs the key LightGBM failure experiment
    ```
2.  **UCI Credit (The Baseline):**
    ```bash
    cd uci_credit
    python train_four_models.py
    ```

***
