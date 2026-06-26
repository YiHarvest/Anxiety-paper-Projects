"""
Step 5: ABIS single-marker model + Bootstrap 95% CI + Core model comparison.
Builds ABIS LR, merges all model results, bootstraps core model AUCs,
and produces final comparison figures and tables.
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
from sklearn.model_selection import StratifiedKFold
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

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
PROJECT = Path(r"D:\sleep\AnxietyProjects")
TRAIN_PATH = PROJECT / "output" / "step2_preprocess_abis" / "train_with_abis_model.csv"
TEST_PATH = PROJECT / "output" / "step2_preprocess_abis" / "test_with_abis_model.csv"
STEP3_PERF = PROJECT / "output" / "step3_single_six_models" / "table3_step3_model_performance.csv"
STEP3_PRED = PROJECT / "output" / "step3_single_six_models" / "step3_test_predictions.csv"
STEP3_BEST = PROJECT / "output" / "step3_single_six_models" / "best_single_biomarker.csv"
STEP4_PERF = PROJECT / "output" / "step4_ratio_integrated_models" / "table5_step4_model_performance.csv"
STEP4_COMB = PROJECT / "output" / "step4_ratio_integrated_models" / "table6_step3_step4_combined_performance.csv"
STEP4_PRED = PROJECT / "output" / "step4_ratio_integrated_models" / "step4_test_predictions.csv"
OUT_DIR = PROJECT / "output" / "step5_abis_bootstrap_compare"

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284
N_BOOTSTRAP = 1000

# =============================================================================
# Helpers
# =============================================================================

def compute_youden_threshold(y_true, y_prob):
    """Youden index = sensitivity + specificity - 1."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values
    idx = np.argsort(y_prob)[::-1]
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


def bootstrap_auc(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, rng=None):
    """Bootstrap 95% CI for ROC_AUC and PR_AUC. Returns dict."""
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    roc_vals = []
    pr_vals = []
    valid_count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        # skip if only one class
        if len(np.unique(yt)) < 2:
            continue
        roc_vals.append(roc_auc_score(yt, yp))
        pr_vals.append(average_precision_score(yt, yp))
        valid_count += 1

    roc_arr = np.array(roc_vals)
    pr_arr = np.array(pr_vals)
    return {
        "ROC_AUC": np.median(roc_arr),
        "ROC_AUC_CI_lower": np.percentile(roc_arr, 2.5),
        "ROC_AUC_CI_upper": np.percentile(roc_arr, 97.5),
        "PR_AUC": np.median(pr_arr),
        "PR_AUC_CI_lower": np.percentile(pr_arr, 2.5),
        "PR_AUC_CI_upper": np.percentile(pr_arr, 97.5),
        "Valid_bootstrap_times": valid_count,
    }


# =============================================================================
# Read data
# =============================================================================
for p in [TRAIN_PATH, TEST_PATH, STEP3_PERF, STEP3_PRED, STEP3_BEST,
          STEP4_PERF, STEP4_PRED]:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for col in [OUTCOME, "ABIS"]:
    if col not in train_df.columns:
        raise ValueError(f"Column '{col}' not in training data")
    if col not in test_df.columns:
        raise ValueError(f"Column '{col}' not in test data")

X_train_abis = train_df[["ABIS"]].copy()
y_train = train_df[OUTCOME].copy()
X_test_abis = test_df[["ABIS"]].copy()
y_test = test_df[OUTCOME].copy()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# =============================================================================
# Load step3/4 data
# =============================================================================
step3_perf = pd.read_csv(STEP3_PERF)
step3_pred = pd.read_csv(STEP3_PRED)
step3_best = pd.read_csv(STEP3_BEST)
step4_perf = pd.read_csv(STEP4_PERF)
step4_pred = pd.read_csv(STEP4_PRED)

# =============================================================================
# ABIS model
# =============================================================================
print("--- ABIS Logistic Regression ---")

# OOF threshold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_abis = np.zeros(len(y_train))
for tr_idx, val_idx in skf.split(X_train_abis, y_train):
    m = LogisticRegression(class_weight="balanced", solver="liblinear",
                           random_state=RANDOM_STATE, max_iter=5000)
    m.fit(X_train_abis.iloc[tr_idx], y_train.iloc[tr_idx])
    oof_abis[val_idx] = m.predict_proba(X_train_abis.iloc[val_idx])[:, 1]
thresh_abis, _ = compute_youden_threshold(y_train, oof_abis)
print(f"  Youden threshold: {thresh_abis:.4f}")

# Final model
abis_model = LogisticRegression(class_weight="balanced", solver="liblinear",
                                random_state=RANDOM_STATE, max_iter=5000)
abis_model.fit(X_train_abis, y_train)
abis_prob = abis_model.predict_proba(X_test_abis)[:, 1]
abis_pred = (abis_prob >= thresh_abis).astype(int)

abis_metrics = evaluate_model(y_test, abis_pred, abis_prob)
abis_metrics.update({
    "Model": "ABIS_LR",
    "Feature_group": "ABIS",
    "Features": "ABIS",
    "Threshold": round(thresh_abis, 4),
})
print(f"  ROC_AUC: {abis_metrics['ROC_AUC']:.4f}, PR_AUC: {abis_metrics['PR_AUC']:.4f}")

# =============================================================================
# Save table8: ABIS performance
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

col_order_perf = [
    "Model", "Feature_group", "Features", "Threshold",
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "TN", "FP", "FN", "TP",
]
tab8 = pd.DataFrame([abis_metrics])[col_order_perf]
tab8.to_csv(OUT_DIR / "table8_abis_model_performance.csv", index=False)
print("Saved table8_abis_model_performance.csv")

# =============================================================================
# Save ABIS test predictions
# =============================================================================
abis_pred_df = pd.DataFrame({
    "sample_index": range(len(test_df)),
    "Anxiety_14": y_test.values,
    "ABIS_predicted_probability": abis_prob,
    "ABIS_predicted_label": abis_pred,
})
abis_pred_df.to_csv(OUT_DIR / "step5_abis_test_predictions.csv", index=False)
print("Saved step5_abis_test_predictions.csv")

# =============================================================================
# Table9: Merge step3 + step4 + step5 all model performance
# =============================================================================
# Harmonise columns: step3/step4 have Best_C, ABIS doesn't. Strip it for merge.
perf_cols = [
    "Model", "Feature_group", "Features", "Threshold",
    "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "TN", "FP", "FN", "TP",
]

def safe_subset(df, cols):
    return df[[c for c in cols if c in df.columns]]

tab9 = pd.concat([
    safe_subset(step3_perf, perf_cols),
    safe_subset(step4_perf, perf_cols),
    safe_subset(tab8, perf_cols),
], ignore_index=True)
tab9 = tab9.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
tab9.to_csv(OUT_DIR / "table9_all_model_performance_ranked.csv", index=False)
print(f"Saved table9_all_model_performance_ranked.csv ({len(tab9)} models)")

# =============================================================================
# Determine which models actually exist for "core model" list
# =============================================================================
best_single_name = step3_best["Best_single_model"].values[0]  # e.g. "LR_CRP"

# Check which models are present in prediction files
all_models_available = set()
for col in step3_pred.columns:
    if col.endswith("_probability"):
        all_models_available.add(col.replace("_probability", ""))
for col in step4_pred.columns:
    if col.endswith("_probability"):
        all_models_available.add(col.replace("_probability", ""))

print(f"\nAvailable models from prediction files: {sorted(all_models_available)}")

# Define core model names and where to find their probabilities
core_candidates = [
    ("Best_Single", best_single_name, step3_pred),
    ("Six_LASSO", "Six_LASSO", step3_pred),
    ("Six_RF", "Six_RF", step3_pred),
    ("Six_XGBoost", "Six_XGBoost", step3_pred),
    ("Ratio_LASSO", "Ratio_LASSO", step4_pred),
    ("Integrated_XGBoost", "Integrated_XGBoost", step4_pred),
    ("ABIS_LR", "ABIS_proba", None),  # special: we compute from model
]

# Build core model mapping: display_name -> (prob_array, source_note)
core_models = {}    # display_name -> (probs, source_note)
core_prob_col_used = {}

for display, col_base, pred_df in core_candidates:
    if display == "ABIS_LR":
        core_models[display] = (abis_prob, "computed from ABIS model")
        core_prob_col_used[display] = "ABIS_predicted_probability (computed)"
        continue
    prob_col = f"{col_base}_probability"
    if prob_col in pred_df.columns:
        core_models[display] = (pred_df[prob_col].values, "from prediction file")
        core_prob_col_used[display] = prob_col
    else:
        print(f"  WARNING: Core model '{display}' (column '{prob_col}') not found – skipping.")

print(f"\nCore models for comparison ({len(core_models)}): {list(core_models.keys())}")

# =============================================================================
# Table10: Core model comparison
# =============================================================================
tab10_rows = []
for display_name in core_models:
    # Map display_name to actual Model name in tab9
    if display_name == "Best_Single":
        match_name = best_single_name
    elif display_name == "ABIS_LR":
        match_name = "ABIS_LR"
    else:
        match_name = display_name
    row = tab9[tab9["Model"] == match_name]
    if len(row) > 0:
        tab10_rows.append(row.iloc[0].to_dict())
    else:
        if match_name == "ABIS_LR":
            tab10_rows.append(abis_metrics)
        else:
            print(f"  WARNING: No performance row for {display_name} (lookup={match_name})")

tab10 = pd.DataFrame(tab10_rows)[perf_cols]
tab10.to_csv(OUT_DIR / "table10_core_model_comparison.csv", index=False)
print(f"Saved table10_core_model_comparison.csv ({len(tab10)} core models)")

# =============================================================================
# Bootstrap 95% CI for core models
# =============================================================================
print("\n--- Bootstrap 95% CI (n=1000) ---")
rng = np.random.RandomState(RANDOM_STATE)
bootstrap_rows = []
for display_name, (probs, source) in core_models.items():
    bs = bootstrap_auc(y_test, probs, N_BOOTSTRAP, rng)
    bs["Model"] = display_name
    bootstrap_rows.append(bs)
    print(f"  {display_name:25s} ROC_AUC={bs['ROC_AUC']:.4f} [{bs['ROC_AUC_CI_lower']:.4f}, {bs['ROC_AUC_CI_upper']:.4f}]  "
          f"PR_AUC={bs['PR_AUC']:.4f}  valid={bs['Valid_bootstrap_times']}")

tab11 = pd.DataFrame(bootstrap_rows)[
    ["Model", "ROC_AUC", "ROC_AUC_CI_lower", "ROC_AUC_CI_upper",
     "PR_AUC", "PR_AUC_CI_lower", "PR_AUC_CI_upper", "Valid_bootstrap_times"]
]
tab11.to_csv(OUT_DIR / "table11_bootstrap_auc_ci.csv", index=False)
print("Saved table11_bootstrap_auc_ci.csv")

# =============================================================================
# Figure 7: Core model ROC curves
# =============================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

n_core = len(core_models)
colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_core)))

fig, ax = plt.subplots(figsize=(7, 7))
for i, (display_name, (probs, _)) in enumerate(core_models.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{display_name} (AUC={auc_val:.3f})",
            color=colors[i % len(colors)], linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("Core Model ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure7_core_model_roc_curves.png", dpi=300)
plt.close(fig)
print("Saved figure7_core_model_roc_curves.png")

# =============================================================================
# Figure 8: Core model PR curves
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 7))
for i, (display_name, (probs, _)) in enumerate(core_models.items()):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    ax.plot(rec, prec, label=f"{display_name} (PR-AUC={pr_auc:.3f})",
            color=colors[i % len(colors)], linewidth=1.5)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Core Model PR Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure8_core_model_pr_curves.png", dpi=300)
plt.close(fig)
print("Saved figure8_core_model_pr_curves.png")

# =============================================================================
# Figure 9: ABIS distribution boxplot + scatter on test set
# =============================================================================
abis0 = test_df.loc[test_df[OUTCOME] == 0, "ABIS"].values
abis1 = test_df.loc[test_df[OUTCOME] == 1, "ABIS"].values

fig, ax = plt.subplots(figsize=(6, 6))
bp = ax.boxplot([abis0, abis1], patch_artist=True, widths=0.4,
                boxprops=dict(facecolor="#b3d9ff", edgecolor="black", linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                medianprops=dict(color="red", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
ax.set_xticklabels(["Anxiety=0", "Anxiety=1"])

# Overlay scatter
for i, vals in enumerate([abis0, abis1]):
    jitter = np.random.RandomState(RANDOM_STATE).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(np.full(len(vals), i + 1) + jitter, vals, alpha=0.4,
               s=20, color="black", edgecolors="none", zorder=3)

ax.set_ylabel("ABIS Score", fontsize=12)
ax.set_title("ABIS Distribution by Anxiety Status (Test Set)", fontsize=13, fontweight="bold")
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure9_test_abis_distribution.png", dpi=300)
plt.close(fig)
print("Saved figure9_test_abis_distribution.png")

# =============================================================================
# Save ABIS model
# =============================================================================
with open(OUT_DIR / "model_abis_logistic.pkl", "wb") as f:
    pickle.dump({"model": abis_model, "threshold": thresh_abis}, f)
print("Saved model_abis_logistic.pkl")

# =============================================================================
# Log
# =============================================================================
best_core_model = max(bootstrap_rows, key=lambda x: x["ROC_AUC"])["Model"]

# ABIS vs best single
abis_row = [r for r in bootstrap_rows if r["Model"] == "ABIS_LR"]
best_single_bs = [r for r in bootstrap_rows if r["Model"] == "Best_Single"]
abis_vs_single = "N/A"
if abis_row and best_single_bs:
    abis_vs_single = "Yes" if abis_row[0]["ROC_AUC"] > best_single_bs[0]["ROC_AUC"] else "No"

# ABIS vs best six-biomarker
six_models_bs = [r for r in bootstrap_rows if "Six" in r["Model"] or "Integrated" in r["Model"]]
best_six_bs = max(six_models_bs, key=lambda x: x["ROC_AUC"]) if six_models_bs else None
abis_vs_six = "N/A"
if abis_row and best_six_bs:
    diff = abis_row[0]["ROC_AUC"] - best_six_bs["ROC_AUC"]
    abis_vs_six = f"Diff={diff:.4f} (ABIS {'>' if diff > 0 else '<'}= Six-biomarker)"

log = []
log.append("=" * 60)
log.append("Step 5 — ABIS + Bootstrap + Core Comparison  LOG")
log.append("=" * 60)
log.append(f"Train: {TRAIN_PATH}  (n={len(train_df)})")
log.append(f"Test:  {TEST_PATH}  (n={len(test_df)})")
log.append(f"ABIS threshold (Youden): {thresh_abis:.4f}")
log.append(f"ABIS test ROC_AUC: {abis_metrics['ROC_AUC']:.4f}")
log.append(f"ABIS test PR_AUC:  {abis_metrics['PR_AUC']:.4f}")
log.append("")
log.append(f"Core models ({len(core_models)}):")
for display_name in core_models:
    log.append(f"  {display_name}: prob_col={core_prob_col_used.get(display_name, 'N/A')}")
log.append("")
log.append("Bootstrap results:")
for bs in bootstrap_rows:
    log.append(f"  {bs['Model']:25s} ROC={bs['ROC_AUC']:.4f} [{bs['ROC_AUC_CI_lower']:.4f}, {bs['ROC_AUC_CI_upper']:.4f}]  "
               f"PR={bs['PR_AUC']:.4f}  valid_n={bs['Valid_bootstrap_times']}")
log.append("")
log.append(f"Best core model (by bootstrap ROC_AUC): {best_core_model}")
log.append(f"ABIS better than best single biomarker?  {abis_vs_single}")
log.append(f"ABIS vs best six-biomarker model:  {abis_vs_six}")
log.append("")
log.append(f"Total models in table9: {len(tab9)}")
log.append("")
log.append("=" * 60)
log.append("Step 5 finished.")
log.append("=" * 60)

with open(OUT_DIR / "step5_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\n".join(log))
print(f"\nStep 5 finished.")
print(f"Results saved to {OUT_DIR}\\")
