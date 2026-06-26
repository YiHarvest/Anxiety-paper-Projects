"""
Step 8: Calibration curves, Brier score, DCA, and 100x repeated random-split
robustness analysis for core anxiety classification models.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

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
STEP3_PRED = PROJECT / "output" / "step3_single_six_models" / "step3_test_predictions.csv"
STEP4_PRED = PROJECT / "output" / "step4_ratio_integrated_models" / "step4_test_predictions.csv"
STEP5_PRED = PROJECT / "output" / "step5_abis_bootstrap_compare" / "step5_abis_test_predictions.csv"
STEP5_CORE = PROJECT / "output" / "step5_abis_bootstrap_compare" / "table10_core_model_comparison.csv"
RAW_TRAIN = PROJECT / "dataset" / "dataset" / "anxiety_bio15_seed284_train_sorted_by_difficulty.csv"
RAW_TEST  = PROJECT / "dataset" / "dataset" / "anxiety_bio15_seed284_test_sorted_by_difficulty.csv"
OUT_DIR   = PROJECT / "output" / "step8_calibration_dca_sensitivity"

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284
N_REPEATS = 1000  # for repeated random split

RAW_BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]
RATIO_FEATURES = [
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "CORT/ACTH", "CORT/IL6", "CORT/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]
INTEGRATED_FEATURES = RAW_BIOMARKERS + RATIO_FEATURES  # 15 features
ALL_PREPROCESS_FEATURES = INTEGRATED_FEATURES  # 15 features to preprocess

# Core models: (display_name, prob_column_map)
# step3 predictions: LR_CRP_probability, Six_XGBoost_probability, Six_RF_probability
# step4 predictions: Ratio_LASSO_probability, Integrated_XGBoost_probability
# step5 predictions: ABIS_predicted_probability
CORE_MODEL_SPECS = [
    ("CRP_only",         "LR_CRP_probability",             STEP3_PRED),
    ("Six_XGBoost",      "Six_XGBoost_probability",        STEP3_PRED),
    ("Six_RF",           "Six_RF_probability",             STEP3_PRED),
    ("Ratio_LASSO",      "Ratio_LASSO_probability",        STEP4_PRED),
    ("Integrated_XGBoost","Integrated_XGBoost_probability", STEP4_PRED),
    ("ABIS_LR",          "ABIS_predicted_probability",     STEP5_PRED),
]


# =============================================================================
# Helpers: preprocessing for repeated split
# =============================================================================

def preprocess_biomarkers(train_df, test_df):
    """
    For ALL_PREPROCESS_FEATURES:
      median impute (train fit) -> log1p -> winsorization 1/99% (train fit)
      -> z-score (train fit).
    Returns (X_train_pp, X_test_pp, pars_dict) for 15 features.
    Also computes ABIS on train and test raw bios.
    """
    X_tr = train_df[ALL_PREPROCESS_FEATURES].copy()
    X_te = test_df[ALL_PREPROCESS_FEATURES].copy()

    # 1. Median imputation
    medians = X_tr.median()
    X_tr = X_tr.fillna(medians)
    X_te = X_te.fillna(medians)

    # 2. log1p (ensuring non-negative)
    X_tr = X_tr.clip(lower=0).apply(np.log1p)
    X_te = X_te.clip(lower=0).apply(np.log1p)

    # 3. Winsorization – fit on train
    lower = X_tr.quantile(0.01)
    upper = X_tr.quantile(0.99)
    X_tr = X_tr.clip(lower=lower, upper=upper, axis=1)
    X_te = X_te.clip(lower=lower, upper=upper, axis=1)

    # 4. Z-score
    scaler = StandardScaler()
    X_tr_arr = scaler.fit_transform(X_tr)
    X_te_arr = scaler.transform(X_te)
    X_tr_pp = pd.DataFrame(X_tr_arr, columns=ALL_PREPROCESS_FEATURES, index=X_tr.index)
    X_te_pp = pd.DataFrame(X_te_arr, columns=ALL_PREPROCESS_FEATURES, index=X_te.index)

    # 5. ABIS from preprocessed raw biomarkers
    def _compute_abis(df_pp):
        return (
            df_pp[["IL6", "TNFalpha", "CRP"]].mean(axis=1)
            - df_pp["IL10"]
            + df_pp["CORT"]
            - df_pp["ACTH"]
        )

    train_abis = _compute_abis(X_tr_pp)
    test_abis  = _compute_abis(X_te_pp)

    return X_tr_pp, X_te_pp, train_abis, test_abis


def train_evaluate_one(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC": average_precision_score(y_test, prob),
        "Brier_score": brier_score_loss(y_test, prob),
        "Accuracy": accuracy_score(y_test, pred),
        "Sensitivity": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "F1": f1_score(y_test, pred, zero_division=0),
    }


# =============================================================================
# DCA helper
# =============================================================================

def net_benefit(y_true, y_prob, thresholds):
    """
    Net benefit = TP/N - FP/N * t/(1-t)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    nb = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb[i] = tp / n - (fp / n) * (t / (1 - t))
    return nb


def treat_all_benefit(y_true, thresholds):
    n = len(y_true)
    prev = y_true.mean()
    nb = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        nb[i] = prev - (1 - prev) * (t / (1 - t))
    return nb


# =============================================================================
# Validate inputs
# =============================================================================
for p in [STEP3_PRED, STEP4_PRED, STEP5_PRED, RAW_TRAIN, RAW_TEST]:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load prediction data
# =============================================================================
print("=== Loading prediction files ===")
pred3 = pd.read_csv(STEP3_PRED)
pred4 = pd.read_csv(STEP4_PRED)
pred5 = pd.read_csv(STEP5_PRED)

# Verify outcome column
for df, src in [(pred3, "step3"), (pred4, "step4"), (pred5, "step5")]:
    if OUTCOME not in df.columns:
        raise ValueError(f"{src} predictions missing '{OUTCOME}' column")

y_test_fixed = pred3[OUTCOME].values  # same for all

# Build lookup: display_name -> probability array
fixed_probs = {}
prob_col_log = {}
for display, prob_col, src_path in CORE_MODEL_SPECS:
    if src_path == STEP3_PRED:
        df = pred3
    elif src_path == STEP4_PRED:
        df = pred4
    else:
        df = pred5
    if prob_col in df.columns:
        fixed_probs[display] = df[prob_col].values
        prob_col_log[display] = f"{prob_col} (from {src_path.name})"
        print(f"  {display}: {prob_col}")
    else:
        print(f"  WARNING: {display}: column '{prob_col}' not found in {src_path.name}")

# =============================================================================
# Part 1: Brier score + Calibration curves
# =============================================================================
print("\n=== Part 1: Brier Score & Calibration ===")

brier_rows = []
for display, probs in fixed_probs.items():
    bs = brier_score_loss(y_test_fixed, probs)
    brier_rows.append({"Model": display, "Brier_score": bs, "N": len(y_test_fixed)})
    print(f"  {display}: Brier={bs:.4f}")

tab20 = pd.DataFrame(brier_rows)
tab20.to_csv(OUT_DIR / "table20_brier_score.csv", index=False)
print("Saved table20_brier_score.csv")

# Calibration curve points
cal_points = []
for n_bins in [5, 10]:
    for display, probs in fixed_probs.items():
        frac_pos, mean_prob = calibration_curve(
            y_test_fixed, probs, n_bins=n_bins, strategy="uniform"
        )
        for i in range(len(frac_pos)):
            cal_points.append({
                "Model": display,
                "n_bins": n_bins,
                "bin_id": i + 1,
                "mean_predicted_probability": mean_prob[i],
                "fraction_of_positives": frac_pos[i],
            })

tab21 = pd.DataFrame(cal_points)
tab21.to_csv(OUT_DIR / "table21_calibration_curve_points.csv", index=False)
print("Saved table21_calibration_curve_points.csv")

# Plot calibration curves
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
colors_cal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for n_bins, fname in [(5, "figure28_calibration_curve_5bins"),
                       (10, "figure29_calibration_curve_10bins")]:
    fig, ax = plt.subplots(figsize=(7, 7))
    sub = tab21[tab21["n_bins"] == n_bins]
    for i, display in enumerate(fixed_probs.keys()):
        pts = sub[sub["Model"] == display]
        if len(pts) == 0:
            continue
        ax.plot(pts["mean_predicted_probability"], pts["fraction_of_positives"],
                "s-", color=colors_cal[i % len(colors_cal)], linewidth=1.2,
                markersize=5, label=display)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Calibration Curves ({n_bins} bins)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=300)
    plt.close(fig)
    print(f"Saved {fname}.png")

# =============================================================================
# Part 2: Decision Curve Analysis
# =============================================================================
print("\n=== Part 2: DCA ===")

thresholds = np.arange(0.01, 1.00, 0.01)
dca_rows = []

treat_all_nb = treat_all_benefit(y_test_fixed, thresholds)
treat_none_nb = np.zeros(len(thresholds))

for t, nb_all, nb_none in zip(thresholds, treat_all_nb, treat_none_nb):
    dca_rows.append({"threshold": round(t, 2), "Model": "Treat_all", "Net_benefit": nb_all})
    dca_rows.append({"threshold": round(t, 2), "Model": "Treat_none", "Net_benefit": nb_none})

for display, probs in fixed_probs.items():
    nb_model = net_benefit(y_test_fixed, probs, thresholds)
    for t, nb in zip(thresholds, nb_model):
        dca_rows.append({"threshold": round(t, 2), "Model": display, "Net_benefit": nb})

tab22 = pd.DataFrame(dca_rows)
tab22.to_csv(OUT_DIR / "table22_dca_net_benefit.csv", index=False)
print("Saved table22_dca_net_benefit.csv")

# DCA plot (threshold 0.05–0.80)
fig, ax = plt.subplots(figsize=(8, 6))
mask = (thresholds >= 0.05) & (thresholds <= 0.80)
t_plot = thresholds[mask]

ax.plot(t_plot, treat_all_nb[mask], "k--", linewidth=1, label="Treat all")
ax.plot(t_plot, treat_none_nb[mask], "k-", linewidth=0.8, label="Treat none", alpha=0.5)

for i, (display, probs) in enumerate(fixed_probs.items()):
    nb_model = net_benefit(y_test_fixed, probs, thresholds)
    ax.plot(t_plot, nb_model[mask], color=colors_cal[i % len(colors_cal)],
            linewidth=1.3, label=display)

ax.set_xlabel("Threshold Probability", fontsize=12)
ax.set_ylabel("Net Benefit", fontsize=12)
ax.set_title("Decision Curve Analysis", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure30_decision_curve_analysis.png", dpi=300)
plt.close(fig)
print("Saved figure30_decision_curve_analysis.png")

# =============================================================================
# Part 3: 100x repeated random-split robustness
# =============================================================================
print("\n=== Part 3: 100x Repeated Random Split ===")

# Merge raw train + test
raw_train = pd.read_csv(RAW_TRAIN)
raw_test  = pd.read_csv(RAW_TEST)
full_raw = pd.concat([raw_train, raw_test], ignore_index=True)
print(f"Full raw dataset: {len(full_raw)} samples")

# Verify required columns exist
required_raw_cols = RAW_BIOMARKERS + RATIO_FEATURES + [OUTCOME]
for col in required_raw_cols:
    if col not in full_raw.columns:
        raise ValueError(f"Column '{col}' not in raw dataset")
# Drop forbidden features if they exist
full_raw = full_raw[required_raw_cols]

y_full = full_raw[OUTCOME].values

repeat_results = []

for rep in range(N_REPEATS):
    seed = RANDOM_STATE + rep
    # Stratified 7:3 split
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # We need a single 7:3 split. Approach: use 3 folds as test, 7 as train
    # Simpler: use train_test_split logic manually
    from sklearn.model_selection import train_test_split
    tr_idx, te_idx = train_test_split(
        np.arange(len(full_raw)), test_size=0.3, stratify=y_full,
        random_state=seed,
    )
    X_full_tr = full_raw.iloc[tr_idx].copy()
    X_full_te = full_raw.iloc[te_idx].copy()
    y_tr = y_full[tr_idx]
    y_te = y_full[te_idx]

    # Preprocess
    X_tr_pp, X_te_pp, abis_tr, abis_te = preprocess_biomarkers(X_full_tr, X_full_te)

    # Models to train
    models_to_train = []

    # A: CRP-only LR
    models_to_train.append((
        "CRP_only",
        LogisticRegression(class_weight="balanced", solver="liblinear",
                           random_state=RANDOM_STATE, max_iter=5000),
        X_tr_pp[["CRP"]], X_te_pp[["CRP"]],
    ))

    # B: Six_RF
    models_to_train.append((
        "Six_RF",
        RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                               min_samples_leaf=3, class_weight="balanced",
                               random_state=RANDOM_STATE),
        X_tr_pp[RAW_BIOMARKERS], X_te_pp[RAW_BIOMARKERS],
    ))

    # C: Six_XGBoost
    if HAS_XGBOOST:
        models_to_train.append((
            "Six_XGBoost",
            xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=RANDOM_STATE,
                              verbosity=0),
            X_tr_pp[RAW_BIOMARKERS], X_te_pp[RAW_BIOMARKERS],
        ))

    # D: Ratio_LASSO
    models_to_train.append((
        "Ratio_LASSO",
        LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                           class_weight="balanced", random_state=RANDOM_STATE,
                           max_iter=5000),
        X_tr_pp[RATIO_FEATURES], X_te_pp[RATIO_FEATURES],
    ))

    # E: Integrated_XGBoost
    if HAS_XGBOOST:
        models_to_train.append((
            "Integrated_XGBoost",
            xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=RANDOM_STATE,
                              verbosity=0),
            X_tr_pp[INTEGRATED_FEATURES], X_te_pp[INTEGRATED_FEATURES],
        ))

    # F: ABIS_LR
    models_to_train.append((
        "ABIS_LR",
        LogisticRegression(class_weight="balanced", solver="liblinear",
                           random_state=RANDOM_STATE, max_iter=5000),
        pd.DataFrame({"ABIS": abis_tr}), pd.DataFrame({"ABIS": abis_te}),
    ))

    for name, model, X_tr_m, X_te_m in models_to_train:
        metrics = train_evaluate_one(model, X_tr_m, y_tr, X_te_m, y_te)
        metrics["repeat_id"] = rep + 1
        metrics["random_state"] = seed
        metrics["Model"] = name
        repeat_results.append(metrics)

    if (rep + 1) % 100 == 0:
        print(f"  Completed {rep + 1}/{N_REPEATS} repeats")

tab23 = pd.DataFrame(repeat_results)
col_order = [
    "repeat_id", "random_state", "Model",
    "ROC_AUC", "PR_AUC", "Brier_score",
    "Accuracy", "Sensitivity", "Specificity", "F1",
]
tab23 = tab23[col_order]
tab23.to_csv(OUT_DIR / "table23_repeated_split_results.csv", index=False)
print(f"Saved table23_repeated_split_results.csv ({len(tab23)} rows)")

# Summary
print("\n=== Computing summary ===")
summary_rows = []
for model_name in tab23["Model"].unique():
    sub = tab23[tab23["Model"] == model_name]
    q1 = sub["ROC_AUC"].quantile(0.25)
    q3 = sub["ROC_AUC"].quantile(0.75)
    summary_rows.append({
        "Model": model_name,
        "ROC_AUC_mean": sub["ROC_AUC"].mean(),
        "ROC_AUC_sd": sub["ROC_AUC"].std(),
        "ROC_AUC_median": sub["ROC_AUC"].median(),
        "ROC_AUC_Q1": q1,
        "ROC_AUC_Q3": q3,
        "PR_AUC_mean": sub["PR_AUC"].mean(),
        "PR_AUC_sd": sub["PR_AUC"].std(),
        "Brier_score_mean": sub["Brier_score"].mean(),
        "Brier_score_sd": sub["Brier_score"].std(),
        "Valid_repeats": len(sub),
    })

tab24 = pd.DataFrame(summary_rows)
tab24.to_csv(OUT_DIR / "table24_repeated_split_summary.csv", index=False)
print("Saved table24_repeated_split_summary.csv")

# Boxplot: ROC-AUC
print("\n=== Plotting boxplots ===")
model_order_roc = tab24.sort_values("ROC_AUC_median", ascending=False)["Model"].tolist()

fig, ax = plt.subplots(figsize=(10, 6))
box_data = [tab23[tab23["Model"] == m]["ROC_AUC"].values for m in model_order_roc]
bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="#b3d9ff", edgecolor="black", linewidth=0.8),
                whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                medianprops=dict(color="red", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
ax.set_xticklabels(model_order_roc, fontsize=9, rotation=20, ha="right")
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("100-Repeated Random Split ROC-AUC Distribution", fontsize=14, fontweight="bold")
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure31_repeated_split_auc_boxplot.png", dpi=300)
plt.close(fig)
print("Saved figure31_repeated_split_auc_boxplot.png")

# Boxplot: PR-AUC
model_order_pr = tab24.sort_values("PR_AUC_mean", ascending=False)["Model"].tolist()

fig, ax = plt.subplots(figsize=(10, 6))
box_data_pr = [tab23[tab23["Model"] == m]["PR_AUC"].values for m in model_order_pr]
bp = ax.boxplot(box_data_pr, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="#ffd9b3", edgecolor="black", linewidth=0.8),
                whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                medianprops=dict(color="red", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
ax.set_xticklabels(model_order_pr, fontsize=9, rotation=20, ha="right")
ax.set_ylabel("PR-AUC", fontsize=12)
ax.set_title("100-Repeated Random Split PR-AUC Distribution", fontsize=14, fontweight="bold")
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure32_repeated_split_prauc_boxplot.png", dpi=300)
plt.close(fig)
print("Saved figure32_repeated_split_prauc_boxplot.png")

# =============================================================================
# Summary Markdown
# =============================================================================
best_model = tab24.sort_values("ROC_AUC_median", ascending=False).iloc[0]
readme = f"""# Step 8: Calibration, DCA & Sensitivity Analysis

## Part 1: Calibration & Brier Score

Brier scores on the fixed test set measure probabilistic calibration
(lower = better calibrated).

## Part 2: Decision Curve Analysis

DCA evaluates clinical net benefit across threshold probabilities.
Models above the "Treat all" and "Treat none" lines provide clinical value.

## Part 3: 100-Repeated Random Split Robustness

### Best model (by median ROC-AUC)
{best_model['Model']}: ROC-AUC mean={best_model['ROC_AUC_mean']:.4f} (SD={best_model['ROC_AUC_sd']:.4f}),
median={best_model['ROC_AUC_median']:.4f} [Q1={best_model['ROC_AUC_Q1']:.4f}, Q3={best_model['ROC_AUC_Q3']:.4f}]

### Key Insight
The 100-repeat analysis confirms model stability and provides empirical confidence
intervals for ROC-AUC and PR-AUC that account for data split variability.

## Output Files

| File | Content |
|------|---------|
| table20_brier_score.csv | Brier scores for core models |
| table21_calibration_curve_points.csv | Calibration curve points (5 & 10 bins) |
| table22_dca_net_benefit.csv | DCA net benefit values |
| table23_repeated_split_results.csv | All 100-repeat results (per model) |
| table24_repeated_split_summary.csv | Summary statistics across repeats |
| figure28_calibration_curve_5bins.png | Calibration curves (5 bins) |
| figure29_calibration_curve_10bins.png | Calibration curves (10 bins) |
| figure30_decision_curve_analysis.png | DCA plot |
| figure31_repeated_split_auc_boxplot.png | ROC-AUC boxplot across 100 splits |
| figure32_repeated_split_prauc_boxplot.png | PR-AUC boxplot across 100 splits |
"""

with open(OUT_DIR / "README_step8_summary.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("Saved README_step8_summary.md")

# =============================================================================
# Log
# =============================================================================
log = []
log.append("=" * 60)
log.append("Step 8 — Calibration, DCA & Sensitivity  LOG")
log.append("=" * 60)
log.append(f"Fixed test N: {len(y_test_fixed)}")
log.append(f"Full dataset N: {len(full_raw)}")
log.append(f"Repeated splits: {N_REPEATS}")
log.append(f"XGBoost available: {HAS_XGBOOST}")
log.append("")
log.append("--- Probability columns used ---")
for k, v in prob_col_log.items():
    log.append(f"  {k}: {v}")
log.append("")
log.append("--- Output files ---")
for f in sorted(OUT_DIR.iterdir()):
    log.append(f"  {f.name}")
log.append("")
log.append("=" * 60)
log.append("Step 8 finished.")
log.append("=" * 60)

with open(OUT_DIR / "step8_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\n".join(log))
print(f"\nStep 8 finished.")
print(f"Results saved to {OUT_DIR}\\")
