"""
Step 3: Single blood biomarker models + Six-biomarker models.
Builds M1 (6 single-biomarker LR models) and M2 (LASSO, RF, XGBoost).
Uses Youden index for threshold selection via 5-fold out-of-fold CV.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Try importing xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
PROJECT = Path(r"D:\sleep\AnxietyProjects")
TRAIN_PATH = PROJECT / "output" / "step2_preprocess_abis" / "train_with_abis_model.csv"
TEST_PATH = PROJECT / "output" / "step2_preprocess_abis" / "test_with_abis_model.csv"
OUT_DIR = PROJECT / "output" / "step3_single_six_models"

# The 6 raw blood biomarkers – must exist in the data
BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]

# Columns we must exclude (ratios, ABIS, non-biomarker columns)
EXCLUDE_COLS = [
    "CaseNumber", "Depression_18", "Chronic_pain", "ABIS",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "CORT/ACTH", "CORT/IL6", "CORT/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284

# =============================================================================
# Helper functions
# =============================================================================

def compute_youden_threshold(y_true, y_prob):
    """Find the threshold that maximises Youden index = sensitivity + specificity - 1."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values

    # Sort descending by predicted probability
    idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[idx]
    y_prob_sorted = y_prob[idx]

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # Cumulative TP and FP at each position (descending probability)
    tp_cum = np.cumsum(y_true_sorted)          # TP at each rank
    fp_cum = np.arange(1, n + 1) - tp_cum      # FP at each rank

    sensitivity = tp_cum / n_pos
    specificity = (n_neg - fp_cum) / n_neg     # TN / n_neg, TN = n_neg - FP
    youden = sensitivity + specificity - 1

    best_idx = np.argmax(youden)
    best_thresh = y_prob_sorted[best_idx]
    return best_thresh, youden[best_idx]


def oof_threshold_selection(model, X, y, n_splits=5):
    """
    Perform StratifiedKFold out-of-fold prediction to select the best threshold
    via Youden index. Returns (oof_probs, best_threshold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y))
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr = y.iloc[train_idx]
        model_clone = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )
        model_clone.fit(X_tr, y_tr)
        oof_prob[valid_idx] = model_clone.predict_proba(X_val)[:, 1]
    best_thresh, _ = compute_youden_threshold(y, oof_prob)
    return oof_prob, best_thresh


def oof_threshold_selection_any(model_func, X, y, n_splits=5):
    """
    General version: build a fresh model via model_func() for each fold,
    return (oof_probs, best_threshold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y))
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr = y.iloc[train_idx]
        m = model_func()
        m.fit(X_tr, y_tr)
        oof_prob[valid_idx] = m.predict_proba(X_val)[:, 1]
    best_thresh, _ = compute_youden_threshold(y, oof_prob)
    return oof_prob, best_thresh


def evaluate_model(y_true, y_pred, y_prob):
    """Return a dict of evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# =============================================================================
# Read data
# =============================================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Verify essential columns
for col in [OUTCOME] + BIOMARKERS:
    if col not in train_df.columns:
        raise ValueError(f"Column '{col}' not found in training data")
    if col not in test_df.columns:
        raise ValueError(f"Column '{col}' not found in test data")

# Feature matrices (only raw biomarkers) and outcome
X_train = train_df[BIOMARKERS].copy()
y_train = train_df[OUTCOME].copy()
X_test = test_df[BIOMARKERS].copy()
y_test = test_df[OUTCOME].copy()

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Outcome prevalence (train): {y_train.mean():.4f}")
print(f"Outcome prevalence (test):  {y_test.mean():.4f}")
print(f"Biomarkers: {BIOMARKERS}")
print(f"XGBoost available: {HAS_XGBOOST}")

# =============================================================================
# Create output directory
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# M1: Single biomarker Logistic Regression models
# ---------------------------------------------------------------------------
print("\n=== M1: Single Biomarker Models ===")
single_results = []
single_models = {}  # feature_name -> trained model
single_thresholds = {}

for feat in BIOMARKERS:
    print(f"  Training single-biomarker LR for: {feat}")
    # Out-of-fold threshold selection
    X_tr_single = X_train[[feat]]
    oof_probs, best_thresh = oof_threshold_selection(None, X_tr_single, y_train)

    # Train final model on full training set
    lr = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    lr.fit(X_tr_single, y_train)

    # Predict on test set
    X_te_single = X_test[[feat]]
    te_prob = lr.predict_proba(X_te_single)[:, 1]
    te_pred = (te_prob >= best_thresh).astype(int)

    metrics = evaluate_model(y_test, te_pred, te_prob)
    metrics.update({
        "Model": f"LR_{feat}",
        "Feature_group": "Single",
        "Features": feat,
        "Best_C": "NA",
        "Threshold": round(best_thresh, 4),
    })
    single_results.append(metrics)
    single_models[feat] = lr
    single_thresholds[feat] = best_thresh
    print(f"    Threshold: {best_thresh:.4f}, ROC_AUC: {metrics['ROC_AUC']:.4f}")

# Find best single biomarker by ROC-AUC
best_single_feat = max(single_results, key=lambda x: x["ROC_AUC"])
best_single_name = best_single_feat["Model"]
best_single_model = single_models[best_single_feat["Features"]]
best_single_threshold = single_thresholds[best_single_feat["Features"]]

print(f"\n  Best single model: {best_single_name} (ROC_AUC={best_single_feat['ROC_AUC']:.4f})")

# Save best single biomarker info
best_single_df = pd.DataFrame([{
    "Best_single_model": best_single_name,
    "Feature": best_single_feat["Features"],
    "ROC_AUC": best_single_feat["ROC_AUC"],
    "PR_AUC": best_single_feat["PR_AUC"],
    "Sensitivity": best_single_feat["Sensitivity"],
    "Specificity": best_single_feat["Specificity"],
    "Threshold": best_single_feat["Threshold"],
}])
best_single_df.to_csv(OUT_DIR / "best_single_biomarker.csv", index=False)
print("  Saved best_single_biomarker.csv")

# ---------------------------------------------------------------------------
# M2: Six-biomarker models
# ---------------------------------------------------------------------------
print("\n=== M2: Six-Biomarker Models ===")
six_results = []
six_model_objs = {}
six_thresholds = {}

# ---- A. LASSO Logistic Regression with GridSearchCV ----
print("  Training Six-biomarker LASSO LR...")

lasso_base = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=5000,
)
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

lasso_cv = GridSearchCV(
    lasso_base,
    param_grid,
    cv=skf,
    scoring="roc_auc",
    refit=True,
)
lasso_cv.fit(X_train, y_train)

best_C = lasso_cv.best_params_["C"]
print(f"    Best C: {best_C}")
print(f"    Best CV ROC_AUC: {lasso_cv.best_score_:.4f}")

# OOF threshold selection for LASSO
def make_lasso():
    return LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=best_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

oof_lasso, thresh_lasso = oof_threshold_selection_any(make_lasso, X_train, y_train)
print(f"    Threshold (Youden): {thresh_lasso:.4f}")

# Final LASSO model (already fitted by GridSearchCV refit)
lasso_final = lasso_cv.best_estimator_
te_prob_lasso = lasso_final.predict_proba(X_test)[:, 1]
te_pred_lasso = (te_prob_lasso >= thresh_lasso).astype(int)

lasso_metrics = evaluate_model(y_test, te_pred_lasso, te_prob_lasso)
lasso_metrics.update({
    "Model": "Six_LASSO",
    "Feature_group": "Six",
    "Features": "IL6+IL10+TNFalpha+CRP+ACTH+CORT",
    "Best_C": best_C,
    "Threshold": round(thresh_lasso, 4),
})
six_results.append(lasso_metrics)
six_model_objs["six_lasso"] = lasso_final
six_thresholds["six_lasso"] = thresh_lasso

# Save LASSO coefficients
coefs = lasso_final.coef_[0]
nz = (np.abs(coefs) > 1e-8).astype(int)
lasso_coef_df = pd.DataFrame({
    "Feature": BIOMARKERS,
    "Coefficient": coefs,
    "Abs_coefficient": np.abs(coefs),
    "Selected_nonzero": nz,
})
lasso_coef_df.to_csv(OUT_DIR / "table4_step3_six_lasso_coefficients.csv", index=False)
print("  Saved table4_step3_six_lasso_coefficients.csv")

# ---- B. Random Forest ----
print("  Training Six-biomarker Random Forest...")

def make_rf():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

oof_rf, thresh_rf = oof_threshold_selection_any(make_rf, X_train, y_train)
print(f"    Threshold (Youden): {thresh_rf:.4f}")

rf_final = make_rf()
rf_final.fit(X_train, y_train)
te_prob_rf = rf_final.predict_proba(X_test)[:, 1]
te_pred_rf = (te_prob_rf >= thresh_rf).astype(int)

rf_metrics = evaluate_model(y_test, te_pred_rf, te_prob_rf)
rf_metrics.update({
    "Model": "Six_RF",
    "Feature_group": "Six",
    "Features": "IL6+IL10+TNFalpha+CRP+ACTH+CORT",
    "Best_C": "NA",
    "Threshold": round(thresh_rf, 4),
})
six_results.append(rf_metrics)
six_model_objs["six_rf"] = rf_final
six_thresholds["six_rf"] = thresh_rf

# ---- C. XGBoost ----
xgboost_ran = False
if HAS_XGBOOST:
    print("  Training Six-biomarker XGBoost...")

    def make_xgb():
        return xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    oof_xgb, thresh_xgb = oof_threshold_selection_any(make_xgb, X_train, y_train)
    print(f"    Threshold (Youden): {thresh_xgb:.4f}")

    xgb_final = make_xgb()
    xgb_final.fit(X_train, y_train)
    te_prob_xgb = xgb_final.predict_proba(X_test)[:, 1]
    te_pred_xgb = (te_prob_xgb >= thresh_xgb).astype(int)

    xgb_metrics = evaluate_model(y_test, te_pred_xgb, te_prob_xgb)
    xgb_metrics.update({
        "Model": "Six_XGBoost",
        "Feature_group": "Six",
        "Features": "IL6+IL10+TNFalpha+CRP+ACTH+CORT",
        "Best_C": "NA",
        "Threshold": round(thresh_xgb, 4),
    })
    six_results.append(xgb_metrics)
    six_model_objs["six_xgboost"] = xgb_final
    six_thresholds["six_xgboost"] = thresh_xgb
    xgboost_ran = True
else:
    print("  XGBoost not installed – skipping.")

# ---------------------------------------------------------------------------
# Combine all results
# ---------------------------------------------------------------------------
all_results = single_results + six_results
perf_df = pd.DataFrame(all_results)

# Reorder columns
col_order = [
    "Model", "Feature_group", "Features", "Best_C", "Threshold",
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "TN", "FP", "FN", "TP",
]
perf_df = perf_df[col_order]
perf_df.to_csv(OUT_DIR / "table3_step3_model_performance.csv", index=False)
print("\n=== Saved table3_step3_model_performance.csv ===")

# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------
pred_df = pd.DataFrame({
    "sample_index": range(len(test_df)),
    "Anxiety_14": y_test.values,
})

# Single-biomarker models
for feat in BIOMARKERS:
    m = single_models[feat]
    th = single_thresholds[feat]
    prob = m.predict_proba(X_test[[feat]])[:, 1]
    pred_df[f"LR_{feat}_probability"] = prob
    pred_df[f"LR_{feat}_label"] = (prob >= th).astype(int)

# Six-biomarker models
for name_key, label in [
    ("six_lasso", "Six_LASSO"),
    ("six_rf", "Six_RF"),
]:
    m = six_model_objs[name_key]
    th = six_thresholds[name_key]
    prob = m.predict_proba(X_test)[:, 1]
    pred_df[f"{label}_probability"] = prob
    pred_df[f"{label}_label"] = (prob >= th).astype(int)

if xgboost_ran:
    m = six_model_objs["six_xgboost"]
    th = six_thresholds["six_xgboost"]
    prob = m.predict_proba(X_test)[:, 1]
    pred_df["Six_XGBoost_probability"] = prob
    pred_df["Six_XGBoost_label"] = (prob >= th).astype(int)

pred_df.to_csv(OUT_DIR / "step3_test_predictions.csv", index=False)
print("Saved step3_test_predictions.csv")

# ---------------------------------------------------------------------------
# Plotting: ROC & PR curves
# ---------------------------------------------------------------------------
# Font setup: Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

# Collect test probabilities for curves
curve_data = {}

# Best single biomarker
best_feat_name = best_single_feat["Features"]
best_prob = single_models[best_feat_name].predict_proba(X_test[[best_feat_name]])[:, 1]
curve_data[best_single_name] = best_prob

# Six-biomarker models
curve_data["Six_LASSO"] = te_prob_lasso
curve_data["Six_RF"] = te_prob_rf
if xgboost_ran:
    curve_data["Six_XGBoost"] = te_prob_xgb

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(7, 7))
for i, (label, prob) in enumerate(curve_data.items()):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})", color=colors[i], linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("ROC Curves — Step 3", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure2_step3_roc_curves.png", dpi=300)
plt.close(fig)
print("Saved figure2_step3_roc_curves.png")

# --- PR Curves ---
fig, ax = plt.subplots(figsize=(7, 7))
for i, (label, prob) in enumerate(curve_data.items()):
    precision, recall, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    ax.plot(recall, precision, label=f"{label} (PR-AUC={pr_auc:.3f})", color=colors[i], linewidth=1.5)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("PR Curves — Step 3", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure3_step3_pr_curves.png", dpi=300)
plt.close(fig)
print("Saved figure3_step3_pr_curves.png")

# ---------------------------------------------------------------------------
# Save models as pickle
# ---------------------------------------------------------------------------
import pickle

# Best single biomarker model
with open(OUT_DIR / "model_best_single_biomarker.pkl", "wb") as f:
    pickle.dump({
        "model": best_single_model,
        "feature": best_single_feat["Features"],
        "threshold": best_single_threshold,
    }, f)

# LASSO
with open(OUT_DIR / "model_six_lasso.pkl", "wb") as f:
    pickle.dump({
        "model": six_model_objs["six_lasso"],
        "threshold": six_thresholds["six_lasso"],
        "best_C": best_C,
    }, f)

# Random Forest
with open(OUT_DIR / "model_six_random_forest.pkl", "wb") as f:
    pickle.dump({
        "model": six_model_objs["six_rf"],
        "threshold": six_thresholds["six_rf"],
    }, f)

# XGBoost
if xgboost_ran:
    with open(OUT_DIR / "model_six_xgboost.pkl", "wb") as f:
        pickle.dump({
            "model": six_model_objs["six_xgboost"],
            "threshold": six_thresholds["six_xgboost"],
        }, f)

print("Saved model .pkl files.")

# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------
log_lines = []
log_lines.append("=" * 60)
log_lines.append("Step 3 — Single & Six Biomarker Models  LOG")
log_lines.append("=" * 60)
log_lines.append(f"Input train:    {TRAIN_PATH}")
log_lines.append(f"Input test:     {TEST_PATH}")
log_lines.append(f"Train samples:  {len(X_train)}")
log_lines.append(f"Test samples:   {len(X_test)}")
log_lines.append(f"Biomarkers used: {BIOMARKERS}")
log_lines.append("")
log_lines.append("--- M1: Single-biomarker LR models ---")
for feat in BIOMARKERS:
    r = next(x for x in single_results if x["Features"] == feat)
    log_lines.append(f"  LR_{feat}: Threshold={r['Threshold']:.4f}, ROC_AUC={r['ROC_AUC']:.4f}, PR_AUC={r['PR_AUC']:.4f}")
log_lines.append("")
log_lines.append("--- M2: Six-biomarker models ---")
log_lines.append(f"  LASSO: best_C={best_C}, Threshold={thresh_lasso:.4f}, ROC_AUC={lasso_metrics['ROC_AUC']:.4f}, PR_AUC={lasso_metrics['PR_AUC']:.4f}")
log_lines.append(f"  RF:    Threshold={thresh_rf:.4f}, ROC_AUC={rf_metrics['ROC_AUC']:.4f}, PR_AUC={rf_metrics['PR_AUC']:.4f}")
if xgboost_ran:
    log_lines.append(f"  XGBoost: Threshold={thresh_xgb:.4f}, ROC_AUC={xgb_metrics['ROC_AUC']:.4f}, PR_AUC={xgb_metrics['PR_AUC']:.4f}")
log_lines.append("")
log_lines.append(f"XGBoost ran successfully: {xgboost_ran}")
log_lines.append(f"Best single biomarker: {best_single_name} (ROC_AUC={best_single_feat['ROC_AUC']:.4f})")
log_lines.append("")
log_lines.append("=" * 60)
log_lines.append("Step 3 finished.")
log_lines.append("=" * 60)

log_text = "\n".join(log_lines)

with open(OUT_DIR / "step3_log.txt", "w", encoding="utf-8") as f:
    f.write(log_text)

print(log_text)

# =============================================================================
# Final print
# =============================================================================
print(f"\nStep 3 finished.")
print(f"Results saved to {OUT_DIR}\\")
