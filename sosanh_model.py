import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

# ==========================================
# LOAD PREDICTION FILES
# ==========================================

rf = pd.read_csv("rf_probs.csv")

xgb = pd.read_csv("xgb_probs.csv")

lgbm = pd.read_csv("lgbm_probs.csv")

svm = pd.read_csv("svm_probs.csv")

ann = pd.read_csv("ann_probs.csv")

# ==========================================
# ROC CURVE COMPARISON
# ==========================================

plt.figure(figsize=(10, 8))

# ==========================================
# RANDOM FOREST
# ==========================================

rf_fpr, rf_tpr, _ = roc_curve(
    rf["y_test"],
    rf["y_prob_rf"]
)

rf_auc = roc_auc_score(
    rf["y_test"],
    rf["y_prob_rf"]
)

plt.plot(
    rf_fpr,
    rf_tpr,
    linewidth=2,
    label=f"Random Forest (AUC = {rf_auc:.4f})"
)

# ==========================================
# XGBOOST
# ==========================================

xgb_fpr, xgb_tpr, _ = roc_curve(
    xgb["y_test"],
    xgb["y_prob_xgb"]
)

xgb_auc = roc_auc_score(
    xgb["y_test"],
    xgb["y_prob_xgb"]
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    linewidth=2,
    label=f"XGBoost (AUC = {xgb_auc:.4f})"
)

# ==========================================
# LIGHTGBM
# ==========================================

lgbm_fpr, lgbm_tpr, _ = roc_curve(
    lgbm["y_test"],
    lgbm["y_prob_lgbm"]
)

lgbm_auc = roc_auc_score(
    lgbm["y_test"],
    lgbm["y_prob_lgbm"]
)

plt.plot(
    lgbm_fpr,
    lgbm_tpr,
    linewidth=2,
    label=f"LightGBM (AUC = {lgbm_auc:.4f})"
)

# ==========================================
# SVM
# ==========================================

svm_fpr, svm_tpr, _ = roc_curve(
    svm["y_test"],
    svm["y_prob_svm"]
)

svm_auc = roc_auc_score(
    svm["y_test"],
    svm["y_prob_svm"]
)

plt.plot(
    svm_fpr,
    svm_tpr,
    linewidth=2,
    label=f"SVM (AUC = {svm_auc:.4f})"
)

# ==========================================
# ANN
# ==========================================

ann_fpr, ann_tpr, _ = roc_curve(
    ann["y_test"],
    ann["y_prob_ann"]
)

ann_auc = roc_auc_score(
    ann["y_test"],
    ann["y_prob_ann"]
)

plt.plot(
    ann_fpr,
    ann_tpr,
    linewidth=2,
    label=f"ANN (AUC = {ann_auc:.4f})"
)

# ==========================================
# RANDOM BASELINE
# ==========================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='black',
    label='Random Guess'
)

# ==========================================
# FINAL SETTINGS
# ==========================================

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison of ML Models")

plt.legend(loc="lower right")

plt.grid(True)

plt.show()