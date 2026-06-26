"""
Step 4: Ratio-based models + Integrated (raw + ratio) models.
Builds 6 models (Ratio-LASSO, Ratio-RF, Ratio-XGBoost, Integrated-LASSO,
Integrated-RF, Integrated-XGBoost).  Youden-index threshold selection via
5-fold OOF CV.  Merges step3 results for combined tables and figures.
"""

import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
STEP3_PERF = PROJECT / "output" / "step3_single_six_models" / "table3_step3_model_performance.csv"
STEP3_PRED = PROJECT / "output" / "step3_single_six_models" / "step3_test_predictions.csv"
OUT_DIR = PROJECT / "output" / "step4_ratio_integrated_models"

# Features
RATIO_FEATURES = [
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "CORT/ACTH", "CORT/IL6", "CORT/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]
RAW_BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]
INTEGRATED_FEATURES = RAW_BIOMARKERS + RATIO_FEATURES

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284

# =============================================================================
# Helpers
# =============================================================================

def compute_youden_threshold(y_true, y_prob):
    """Youden index = sensitivity + specificity - 1, maximised over thresholds."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values
    idx = np.argsort(y_prob)[::-1]  # descending
    y_true_sorted = y_true[idx]
    y_prob_sorted = y_prob[idx]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.arange(1, len(y_true) + 1) - tp_cum
    sensitivity = tp_cum / n_pos
    specificity = (n_neg - fp_cum) / n_neg
    youden = sensitivity + specificity - 1
    best_idx = np.argmax(youden)
    return y_prob_sorted[best_idx], youden[best_idx]


def oof_threshold_selection(model_func, X, y, n_splits=5):
    """5-fold OOF prediction -> Youden-optimal threshold. Returns (oof_probs, best_thresh)."""
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# =============================================================================
# Read data
# =============================================================================
for p in [TRAIN_PATH, TEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for col in [OUTCOME] + RAW_BIOMARKERS + RATIO_FEATURES:
    if col not in train_df.columns:
        raise ValueError(f"Column '{col}' not in training data")
    if col not in test_df.columns:
        raise ValueError(f"Column '{col}' not in test data")

X_train_ratio = train_df[RATIO_FEATURES].copy()
X_train_integrated = train_df[INTEGRATED_FEATURES].copy()
y_train = train_df[OUTCOME].copy()

X_test_ratio = test_df[RATIO_FEATURES].copy()
X_test_integrated = test_df[INTEGRATED_FEATURES].copy()
y_test = test_df[OUTCOME].copy()

print(f"Train: {len(X_train_ratio)}, Test: {len(X_test_ratio)}")
print(f"XGBoost available: {HAS_XGBOOST}")

# Read step3 results (may fail gracefully for combining)
step3_perf_df = None
step3_merged_ok = False
try:
    if STEP3_PERF.exists() and STEP3_PRED.exists():
        step3_perf_df = pd.read_csv(STEP3_PERF)
        step3_pred_df = pd.read_csv(STEP3_PRED)
        step3_merged_ok = True
        print("Loaded step3 performance and predictions for merging.")
    else:
        print("Step3 files missing – will only output step4 results.")
except Exception as e:
    print(f"Could not read step3 files: {e}")

# =============================================================================
# Output directory
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model builders
# =============================================================================

def make_lasso(C=1.0):
    return LogisticRegression(
        penalty="l1", solver="liblinear",
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000,
        C=C,
    )

def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=5,
        min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE,
    )

def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0,
    )

# =============================================================================
# Build models
# =============================================================================
results = []            # list of metric dicts
model_registry = {}     # name -> (model_obj, threshold)
lasso_coefs = []        # for table7

model_configs = [
    ("Ratio_LASSO",   "Ratio",      X_train_ratio,      RATIO_FEATURES),
    ("Ratio_RF",      "Ratio",      X_train_ratio,      RATIO_FEATURES),
    ("Integrated_LASSO", "Integrated", X_train_integrated, INTEGRATED_FEATURES),
    ("Integrated_RF", "Integrated", X_train_integrated, INTEGRATED_FEATURES),
]

if HAS_XGBOOST:
    model_configs += [
        ("Ratio_XGBoost",      "Ratio",      X_train_ratio,      RATIO_FEATURES),
        ("Integrated_XGBoost", "Integrated", X_train_integrated, INTEGRATED_FEATURES),
    ]

for name, feat_group, X_tr, feat_list in model_configs:
    print(f"\n--- {name} ---")

    if name.endswith("_LASSO"):
        # GridSearchCV for C
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            make_lasso(), {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=skf, scoring="roc_auc", refit=True,
        )
        gs.fit(X_tr, y_train)
        best_C = gs.best_params_["C"]
        print(f"  Best C: {best_C}, CV ROC_AUC: {gs.best_score_:.4f}")

        # OOF threshold
        def model_func():
            return make_lasso(C=best_C)
        _, best_thresh = oof_threshold_selection(model_func, X_tr, y_train)

        # Final model (already refit by GridSearchCV)
        final_model = gs.best_estimator_

        # LASSO coefficients
        coefs = final_model.coef_[0]
        for f, c in zip(feat_list, coefs):
            lasso_coefs.append({
                "Model": name, "Feature_group": feat_group,
                "Feature": f, "Coefficient": c,
                "Abs_coefficient": abs(c),
                "Selected_nonzero": int(abs(c) > 1e-8),
            })
        nz_count = int((np.abs(coefs) > 1e-8).sum())
        print(f"  Non-zero coefficients: {nz_count}")

    elif name.endswith("_RF"):
        best_C = "NA"
        def model_func():
            return make_rf()
        _, best_thresh = oof_threshold_selection(model_func, X_tr, y_train)
        final_model = make_rf()
        final_model.fit(X_tr, y_train)

    elif name.endswith("_XGBoost"):
        best_C = "NA"
        def model_func():
            return make_xgb()
        _, best_thresh = oof_threshold_selection(model_func, X_tr, y_train)
        final_model = make_xgb()
        final_model.fit(X_tr, y_train)

    print(f"  Threshold (Youden): {best_thresh:.4f}")

    # Evaluate on test set
    X_te = test_df[feat_list]
    te_prob = final_model.predict_proba(X_te)[:, 1]
    te_pred = (te_prob >= best_thresh).astype(int)

    metrics = evaluate_model(y_test, te_pred, te_prob)
    metrics.update({
        "Model": name, "Feature_group": feat_group,
        "Features": "+".join(feat_list),
        "Best_C": best_C, "Threshold": round(best_thresh, 4),
    })
    results.append(metrics)
    model_registry[name] = (final_model, best_thresh)
    print(f"  ROC_AUC: {metrics['ROC_AUC']:.4f}, PR_AUC: {metrics['PR_AUC']:.4f}")

# =============================================================================
# Save table5 – step4 performance
# =============================================================================
col_order = [
    "Model", "Feature_group", "Features", "Best_C", "Threshold",
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "TN", "FP", "FN", "TP",
]
perf_step4 = pd.DataFrame(results)[col_order]
perf_step4.to_csv(OUT_DIR / "table5_step4_model_performance.csv", index=False)
print("\nSaved table5_step4_model_performance.csv")

# =============================================================================
# Save table6 – combined step3 + step4, sorted by ROC_AUC desc
# =============================================================================
if step3_perf_df is not None:
    # Ensure both have same columns
    for c in col_order:
        if c not in step3_perf_df.columns:
            step3_perf_df[c] = "NA"
    step3_perf_df = step3_perf_df[col_order]
    combined = pd.concat([step3_perf_df, perf_step4], ignore_index=True)
    combined = combined.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    combined.to_csv(OUT_DIR / "table6_step3_step4_combined_performance.csv", index=False)
    print("Saved table6_step3_step4_combined_performance.csv")
else:
    # Fallback: just step4 sorted
    perf_step4.sort_values("ROC_AUC", ascending=False).to_csv(
        OUT_DIR / "table6_step3_step4_combined_performance.csv", index=False)
    print("Saved table6_step3_step4_combined_performance.csv (step4 only)")

# =============================================================================
# Save step4 test predictions
# =============================================================================
pred_df = pd.DataFrame({
    "sample_index": range(len(test_df)),
    "Anxiety_14": y_test.values,
})

for name, (mdl, th) in model_registry.items():
    feat_list = RATIO_FEATURES if name.startswith("Ratio_") else INTEGRATED_FEATURES
    X_te = test_df[feat_list]
    prob = mdl.predict_proba(X_te)[:, 1]
    pred_df[f"{name}_probability"] = prob
    pred_df[f"{name}_label"] = (prob >= th).astype(int)

pred_df.to_csv(OUT_DIR / "step4_test_predictions.csv", index=False)
print("Saved step4_test_predictions.csv")

# =============================================================================
# Save table7 – LASSO coefficients
# =============================================================================
lasso_df = pd.DataFrame(lasso_coefs)[
    ["Model", "Feature_group", "Feature", "Coefficient", "Abs_coefficient", "Selected_nonzero"]
]
lasso_df.to_csv(OUT_DIR / "table7_step4_lasso_coefficients.csv", index=False)
print("Saved table7_step4_lasso_coefficients.csv")

# =============================================================================
# Save models as pickle
# =============================================================================
for name, (mdl, th) in model_registry.items():
    pkl_name = f"model_{name.lower()}.pkl"
    with open(OUT_DIR / pkl_name, "wb") as f:
        pickle.dump({"model": mdl, "threshold": th}, f)
print("Saved model .pkl files")

# =============================================================================
# Plotting: ROC & PR curves (merge step3 + step4)
# =============================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

# Build curve_data dict: label -> array of probabilities (on test set)
curve_data = {}

# Step4 models
for name, (mdl, th) in model_registry.items():
    feat_list = RATIO_FEATURES if name.startswith("Ratio_") else INTEGRATED_FEATURES
    curve_data[name] = mdl.predict_proba(test_df[feat_list])[:, 1]

# Step3 models – extract probability columns from step3 predictions
merged_step3 = False
if step3_merged_ok:
    merged_step3 = True
    prob_cols = [c for c in step3_pred_df.columns if c.endswith("_probability")]
    for col in prob_cols:
        lbl = col.replace("_probability", "")
        curve_data[f"Step3_{lbl}"] = step3_pred_df[col].values

# Color palette – enough distinct colours
colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(curve_data))))

# --- ROC ---
fig, ax = plt.subplots(figsize=(8, 8))
for i, (label, prob) in enumerate(curve_data.items()):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})",
            color=colors[i % len(colors)], linewidth=1.2)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("ROC Curves — Step 3 + Step 4", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure4_step4_roc_curves.png", dpi=300)
plt.close(fig)
print("Saved figure4_step4_roc_curves.png")

# --- PR ---
fig, ax = plt.subplots(figsize=(8, 8))
for i, (label, prob) in enumerate(curve_data.items()):
    prec, rec, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    ax.plot(rec, prec, label=f"{label} (PR-AUC={pr_auc:.3f})",
            color=colors[i % len(colors)], linewidth=1.2)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("PR Curves — Step 3 + Step 4", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure5_step4_pr_curves.png", dpi=300)
plt.close(fig)
print("Saved figure5_step4_pr_curves.png")

# --- LASSO coefficient bar plot ---
lasso_sub = lasso_df[lasso_df["Abs_coefficient"] > 1e-8].copy()
if len(lasso_sub) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    # Group by model, then feature
    lasso_sub = lasso_sub.copy()
    lasso_sub["label"] = lasso_sub["Model"].str.replace("_", " ") + " | " + lasso_sub["Feature"]
    colors_bar = ["#d62728" if c < 0 else "#1f77b4" for c in lasso_sub["Coefficient"]]
    y_pos = range(len(lasso_sub))
    ax.barh(y_pos, lasso_sub["Coefficient"].values, color=colors_bar, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lasso_sub["label"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Coefficient", fontsize=12)
    ax.set_title("Non-zero LASSO Coefficients — Step 4", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure6_step4_lasso_coefficients.png", dpi=300)
    plt.close(fig)
    print("Saved figure6_step4_lasso_coefficients.png")
else:
    print("All LASSO coefficients are zero – skipping coefficient bar plot.")

# =============================================================================
# Log
# =============================================================================
log = []
log.append("=" * 60)
log.append("Step 4 — Ratio & Integrated Models  LOG")
log.append("=" * 60)
log.append(f"Train: {TRAIN_PATH}  (n={len(train_df)})")
log.append(f"Test:  {TEST_PATH}  (n={len(test_df)})")
log.append(f"Ratio features ({len(RATIO_FEATURES)}): {RATIO_FEATURES}")
log.append(f"Integrated features ({len(INTEGRATED_FEATURES)}): {INTEGRATED_FEATURES}")
log.append(f"XGBoost: {HAS_XGBOOST}")
log.append(f"Merged step3 results: {merged_step3}")
log.append("")
for r in results:
    log.append(f"  {r['Model']}: C={r['Best_C']}, Thresh={r['Threshold']:.4f}, "
               f"ROC_AUC={r['ROC_AUC']:.4f}, PR_AUC={r['PR_AUC']:.4f}")
log.append("")
# Non-zero counts
for m in ["Ratio_LASSO", "Integrated_LASSO"]:
    sub = lasso_df[lasso_df["Model"] == m]
    nz = int(sub["Selected_nonzero"].sum())
    log.append(f"  {m} non-zero coefficients: {nz}")
log.append("")
log.append("=" * 60)
log.append("Step 4 finished.")
log.append("=" * 60)

with open(OUT_DIR / "step4_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\n".join(log))

print(f"\nStep 4 finished.")
print(f"Results saved to {OUT_DIR}\\")
