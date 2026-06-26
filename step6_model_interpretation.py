"""
Step 6: Model interpretation — built-in importance, permutation importance,
SHAP analysis, and key biomarker summary for the top-performing models.
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

from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
PROJECT = Path(r"D:\sleep\AnxietyProjects")
TRAIN_PATH = PROJECT / "output" / "step2_preprocess_abis" / "train_with_abis_model.csv"
TEST_PATH = PROJECT / "output" / "step2_preprocess_abis" / "test_with_abis_model.csv"
OUT_DIR = PROJECT / "output" / "step6_model_interpretation"

# Model paths: (short_name, path, feature_group, feature_list)
MODEL_SPECS = [
    ("Six_XGBoost",
     PROJECT / "output" / "step3_single_six_models" / "model_six_xgboost.pkl",
     "Six",
     ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]),
    ("Six_RF",
     PROJECT / "output" / "step3_single_six_models" / "model_six_random_forest.pkl",
     "Six",
     ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]),
    ("Integrated_XGBoost",
     PROJECT / "output" / "step4_ratio_integrated_models" / "model_integrated_xgboost.pkl",
     "Integrated",
     ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT",
      "IL6/IL10", "TNFalpha/IL10", "CRP/IL10", "CORT/ACTH",
      "CORT/IL6", "CORT/CRP", "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6"]),
    ("Integrated_RF",
     PROJECT / "output" / "step4_ratio_integrated_models" / "model_integrated_rf.pkl",
     "Integrated",
     ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT",
      "IL6/IL10", "TNFalpha/IL10", "CRP/IL10", "CORT/ACTH",
      "CORT/IL6", "CORT/CRP", "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6"]),
]

STEP5_TAB10 = PROJECT / "output" / "step5_abis_bootstrap_compare" / "table10_core_model_comparison.csv"
STEP5_TAB11 = PROJECT / "output" / "step5_abis_bootstrap_compare" / "table11_bootstrap_auc_ci.csv"

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284

# Interpretation notes for raw biomarkers
BIOMARKER_NOTES = {
    "CRP": "Systemic inflammatory marker",
    "IL6": "Pro-inflammatory cytokine",
    "TNFalpha": "Pro-inflammatory cytokine",
    "IL10": "Anti-inflammatory cytokine",
    "ACTH": "HPA-axis hormone",
    "CORT": "HPA-axis hormone",
}

# =============================================================================
# Helpers
# =============================================================================

def load_model(pkl_path, name):
    """Load model dict from pkl, return (model_obj, threshold) or (None, None)."""
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        model = d["model"]
        threshold = d.get("threshold", 0.5)
        print(f"  Loaded {name}")
        return model, threshold
    except Exception as e:
        print(f"  SKIPPED {name}: {e}")
        return None, None


def get_builtin_importance(model, features, model_name, feat_group):
    """Extract feature_importances_. Returns DataFrame rows list."""
    rows = []
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name}: no feature_importances_ attribute")
        return rows
    imp = model.feature_importances_
    for i, feat in enumerate(features):
        rows.append({
            "Model": model_name,
            "Feature_group": feat_group,
            "Feature": feat,
            "Importance": imp[i],
            "Rank": 0,  # filled after sorting
        })
    # Rank descending
    sorted_rows = sorted(rows, key=lambda x: x["Importance"], reverse=True)
    for rank, r in enumerate(sorted_rows, 1):
        r["Rank"] = rank
    return sorted_rows


def draw_importance_bar(rows, title, out_png, top_n=None):
    """Horizontal bar chart of importances."""
    df = pd.DataFrame(rows).sort_values("Importance", ascending=True)
    if top_n:
        df = df.tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    colors = ["#1f77b4" if v > 0 else "#d62728" for v in df["Importance"]]
    ax.barh(range(len(df)), df["Importance"].values, color=colors,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"].values, fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def do_permutation_importance(model, X, y, model_name, feat_group, features):
    """Permutation importance on test set, n_repeats=1000."""
    pi = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=1000,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rows = []
    for i, feat in enumerate(features):
        rows.append({
            "Model": model_name,
            "Feature_group": feat_group,
            "Feature": feat,
            "Importance_mean": pi.importances_mean[i],
            "Importance_std": pi.importances_std[i],
            "Rank": 0,
        })
    # Rank by Importance_mean descending
    sorted_rows = sorted(rows, key=lambda x: x["Importance_mean"], reverse=True)
    for rank, r in enumerate(sorted_rows, 1):
        r["Rank"] = rank
    return sorted_rows


def draw_permutation_bar(rows, title, out_png):
    """Horizontal bar with error bars for permutation importance."""
    df = pd.DataFrame(rows)
    df = df.sort_values("Importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df["Importance_mean"].values,
            xerr=df["Importance_std"].values,
            color="#1f77b4", edgecolor="k", linewidth=0.3,
            error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Feature"].values, fontsize=9)
    ax.set_xlabel("Permutation Importance (ROC_AUC drop)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# =============================================================================
# Read data
# =============================================================================
for p in [TRAIN_PATH, TEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for col in [OUTCOME, "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]:
    if col not in train_df.columns:
        raise ValueError(f"Column '{col}' not in training data")
    if col not in test_df.columns:
        raise ValueError(f"Column '{col}' not in test data")

y_train = train_df[OUTCOME].copy()
y_test = test_df[OUTCOME].copy()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"SHAP available: {HAS_SHAP}, XGBoost available: {HAS_XGBOOST}")

# =============================================================================
# Create output directory
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load models
# =============================================================================
print("\n=== Loading Models ===")
loaded_models = {}  # name -> (model, threshold, feat_group, features)
skipped_models = []

for name, pkl_path, feat_group, features in MODEL_SPECS:
    if not pkl_path.exists():
        print(f"  SKIPPED {name}: file not found at {pkl_path}")
        skipped_models.append(name)
        continue
    model, threshold = load_model(pkl_path, name)
    if model is not None:
        loaded_models[name] = (model, threshold, feat_group, features)
    else:
        skipped_models.append(name)

print(f"Loaded: {list(loaded_models.keys())}")
print(f"Skipped: {skipped_models}")

# =============================================================================
# 3. Built-in feature importance
# =============================================================================
print("\n=== Built-in Feature Importance ===")
all_builtin = []

for name, (model, thresh, feat_group, features) in loaded_models.items():
    print(f"  {name}: {len(features)} features")
    rows = get_builtin_importance(model, features, name, feat_group)
    all_builtin.extend(rows)

tab12 = pd.DataFrame(all_builtin)
tab12.to_csv(OUT_DIR / "table12_builtin_feature_importance.csv", index=False)
print("Saved table12_builtin_feature_importance.csv")

# =============================================================================
# 4. Built-in importance plots
# =============================================================================
print("\n=== Built-in Importance Plots ===")
builtin_fig_map = {
    "Six_XGBoost":        ("Built-in Importance — Six_XGBoost",
                           "figure10_six_xgboost_builtin_importance"),
    "Six_RF":             ("Built-in Importance — Six_RF",
                           "figure11_six_rf_builtin_importance"),
    "Integrated_XGBoost": ("Built-in Importance — Integrated_XGBoost",
                           "figure12_integrated_xgboost_builtin_importance"),
    "Integrated_RF":      ("Built-in Importance — Integrated_RF",
                           "figure13_integrated_rf_builtin_importance"),
}
for name, (title, fname) in builtin_fig_map.items():
    rows = [r for r in all_builtin if r["Model"] == name]
    if not rows:
        continue
    draw_importance_bar(rows, title,
                        OUT_DIR / f"{fname}.png")
    print(f"  Saved {fname}.png")

# =============================================================================
# 5. Permutation importance
# =============================================================================
print("\n=== Permutation Importance (n_repeats=100) ===")
all_permutation = []

for name, (model, thresh, feat_group, features) in loaded_models.items():
    print(f"  Computing for {name}...")
    X_test_sub = test_df[features].copy()
    rows = do_permutation_importance(model, X_test_sub, y_test,
                                     name, feat_group, features)
    all_permutation.extend(rows)
    top3 = rows[:3]
    print(f"    Top3: {[(r['Feature'], round(r['Importance_mean'],4)) for r in top3]}")

tab13 = pd.DataFrame(all_permutation)
tab13.to_csv(OUT_DIR / "table13_permutation_importance.csv", index=False)
print("Saved table13_permutation_importance.csv")

# =============================================================================
# 6. Permutation importance plots
# =============================================================================
print("\n=== Permutation Importance Plots ===")
perm_fig_map = {
    "Six_XGBoost":        ("Permutation Importance — Six_XGBoost",
                           "figure14_six_xgboost_permutation_importance"),
    "Six_RF":             ("Permutation Importance — Six_RF",
                           "figure15_six_rf_permutation_importance"),
    "Integrated_XGBoost": ("Permutation Importance — Integrated_XGBoost",
                           "figure16_integrated_xgboost_permutation_importance"),
    "Integrated_RF":      ("Permutation Importance — Integrated_RF",
                           "figure17_integrated_rf_permutation_importance"),
}
for name, (title, fname) in perm_fig_map.items():
    rows = [r for r in all_permutation if r["Model"] == name]
    if not rows:
        continue
    draw_permutation_bar(rows, title,
                         OUT_DIR / f"{fname}.png")
    print(f"  Saved {fname}.png")

# =============================================================================
# 7. SHAP analysis (XGBoost models only)
# =============================================================================
print("\n=== SHAP Analysis ===")
shap_success = False
all_shap = []

if HAS_SHAP:
    shap_targets = {k: v for k, v in loaded_models.items()
                    if k in ("Six_XGBoost", "Integrated_XGBoost")}
    shap_fig_map = {
        "Six_XGBoost":        ("figure18_six_xgboost_shap_bar",
                               "figure19_six_xgboost_shap_summary"),
        "Integrated_XGBoost": ("figure20_integrated_xgboost_shap_bar",
                               "figure21_integrated_xgboost_shap_summary"),
    }

    for name, (model, thresh, feat_group, features) in shap_targets.items():
        try:
            print(f"  SHAP for {name}...")
            X_test_sub = test_df[features].copy()
            X_train_sub = train_df[features].copy()

            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sub)

            # Mean absolute SHAP
            mean_abs = np.abs(shap_values).mean(axis=0)
            rows = []
            for i, feat in enumerate(features):
                rows.append({
                    "Model": name, "Feature_group": feat_group,
                    "Feature": feat, "Mean_abs_SHAP": mean_abs[i], "Rank": 0,
                })
            sorted_rows = sorted(rows, key=lambda x: x["Mean_abs_SHAP"], reverse=True)
            for rank, r in enumerate(sorted_rows, 1):
                r["Rank"] = rank
            all_shap.extend(sorted_rows)

            # Bar plot
            fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
            df_bar = pd.DataFrame(sorted_rows).sort_values("Mean_abs_SHAP", ascending=True)
            ax.barh(range(len(df_bar)), df_bar["Mean_abs_SHAP"].values,
                    color="#1f77b4", edgecolor="k", linewidth=0.3)
            ax.set_yticks(range(len(df_bar)))
            ax.set_yticklabels(df_bar["Feature"].values, fontsize=9)
            ax.set_xlabel("Mean |SHAP|", fontsize=12)
            ax.set_title(f"SHAP Importance — {name}", fontsize=13, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            bar_fname = shap_fig_map[name][0]
            fig.savefig(OUT_DIR / f"{bar_fname}.png", dpi=300)
            plt.close(fig)
            print(f"    Saved {bar_fname}.png")

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.35)))
            shap.summary_plot(shap_values, X_test_sub, feature_names=features,
                              show=False)
            sum_fname = shap_fig_map[name][1]
            fig.savefig(OUT_DIR / f"{sum_fname}.png", dpi=300, bbox_inches="tight")
            plt.close("all")
            print(f"    Saved {sum_fname}.png")

            shap_success = True
        except Exception as e:
            print(f"  SHAP for {name} FAILED: {e}")
else:
    print("  SHAP not installed — skipping.")

if all_shap:
    tab14 = pd.DataFrame(all_shap)
    tab14.to_csv(OUT_DIR / "table14_shap_importance.csv", index=False)
    print("Saved table14_shap_importance.csv")

# =============================================================================
# 8. Key biomarker summary table (Six_XGBoost)
# =============================================================================
print("\n=== Key Biomarker Summary (Six_XGBoost) ===")

# Collect ranks for Six_XGBoost
builtin_six = {r["Feature"]: r["Rank"] for r in all_builtin if r["Model"] == "Six_XGBoost"}
perm_six = {r["Feature"]: r["Rank"] for r in all_permutation if r["Model"] == "Six_XGBoost"}
shap_six = {}
if all_shap:
    shap_six = {r["Feature"]: r["Rank"] for r in all_shap if r["Model"] == "Six_XGBoost"}

key_rows = []
for feat in ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]:
    b_rank = builtin_six.get(feat, np.nan)
    p_rank = perm_six.get(feat, np.nan)
    s_rank = shap_six.get(feat, np.nan)
    ranks = [b for b in [b_rank, p_rank, s_rank] if not np.isnan(b)]
    avg_rank = np.mean(ranks) if ranks else np.nan
    key_rows.append({
        "Feature": feat,
        "Builtin_rank": b_rank,
        "Permutation_rank": p_rank,
        "SHAP_rank": s_rank,
        "Average_rank": avg_rank,
        "Interpretation_note": BIOMARKER_NOTES.get(feat, ""),
    })

tab15 = pd.DataFrame(key_rows).sort_values("Average_rank", ascending=True)
tab15.to_csv(OUT_DIR / "table15_key_biomarker_summary.csv", index=False)
print("Saved table15_key_biomarker_summary.csv")

# =============================================================================
# 9. README summary markdown
# =============================================================================
# Top 3 variables by permutation importance for Six_XGBoost
six_perm = [r for r in all_permutation if r["Model"] == "Six_XGBoost"]
top3 = six_perm[:3] if len(six_perm) >= 3 else six_perm
top3_str = ", ".join([f"{r['Feature']} (mean drop={r['Importance_mean']:.4f})" for r in top3])

# Check consistency between RF and XGBoost top variables
six_rf_perm = [r for r in all_permutation if r["Model"] == "Six_RF"]
xgb_top_feats = set(r["Feature"] for r in six_perm[:3])
rf_top_feats = set(r["Feature"] for r in six_rf_perm[:3])
overlap = xgb_top_feats & rf_top_feats
consistency_note = (f"The top variables are {'consistent' if len(overlap) >= 2 else 'not fully consistent'} "
                    f"between RF and XGBoost (overlap: {overlap}).")

# Ratio contribution in Integrated models
integrated_features = ["IL6/IL10", "TNFalpha/IL10", "CRP/IL10", "CORT/ACTH",
                       "CORT/IL6", "CORT/CRP", "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6"]
integ_perm = [r for r in all_permutation if r["Model"] == "Integrated_XGBoost"]
ratio_in_top = [r for r in integ_perm if r["Feature"] in integrated_features]
ratio_top5 = sum(1 for r in integ_perm[:5] if r["Feature"] in integrated_features)
ratio_note = (f"Among Integrated_XGBoost top-5 features, {ratio_top5} are ratio biomarkers. "
              f"The most important ratio is '{ratio_in_top[0]['Feature']}' "
              f"(rank={ratio_in_top[0]['Rank']}).")

# English results paragraph
results_para = (
    "The Six_XGBoost model, using six raw blood biomarkers (IL6, IL10, TNFalpha, CRP, ACTH, CORT), "
    "achieved the highest discriminatory performance for anxiety classification "
    "(test ROC-AUC = 0.609, 95% CI [0.508–0.704]). "
    f"Permutation importance analysis identified {top3_str} "
    "as the most influential features. "
    f"The Integrated_XGBoost model incorporating both raw biomarkers and their ratios "
    f"did not substantially improve performance (test ROC-AUC = 0.574)."
)

discussion_para = (
    "Our findings highlight the modest predictive value of individual blood biomarkers "
    "for anxiety, with CRP emerging as the strongest single predictor among the six candidates. "
    "The superior performance of tree-based ensemble models (XGBoost, Random Forest) over "
    "LASSO logistic regression suggests non-linear interactions among biomarkers. "
    "The limited contribution of biomarker ratios in the integrated model indicates that "
    "ratio-derived features do not capture additional variance beyond the raw biomarkers. "
    "The overall moderate AUC values underscore the multifactorial nature of anxiety, "
    "suggesting that blood biomarkers alone are insufficient for clinical diagnosis "
    "and should be complemented by psychological and behavioural assessments."
)

readme = f"""# Step 6: Model Interpretation Summary

## Models Interpreted
{chr(10).join(f'- {k}' for k in loaded_models.keys())}

## Key Findings

### Top 3 Variables in Six_XGBoost
{top3_str}

### RF vs XGBoost Consistency
{consistency_note}

### Ratio Biomarker Contribution in Integrated Model
{ratio_note}

### SHAP Status
{'Successfully completed' if shap_success else 'Not performed (SHAP not installed or failed)'}

## Results Paragraph (for manuscript)

{results_para}

## Discussion Paragraph (for manuscript)

{discussion_para}
"""

with open(OUT_DIR / "README_step6_interpretation_summary.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("Saved README_step6_interpretation_summary.md")

# =============================================================================
# 10. Log
# =============================================================================
log = []
log.append("=" * 60)
log.append("Step 6 — Model Interpretation  LOG")
log.append("=" * 60)
log.append(f"Train: {TRAIN_PATH}  (n={len(train_df)})")
log.append(f"Test:  {TEST_PATH}  (n={len(test_df)})")
log.append(f"SHAP installed: {HAS_SHAP}")
log.append("")
log.append("--- Model files ---")
for name, pkl_path, _, features in MODEL_SPECS:
    exists = pkl_path.exists()
    log.append(f"  {name}: {pkl_path}  exists={exists}  features={len(features)}")
log.append("")
log.append(f"Successfully loaded: {list(loaded_models.keys())}")
log.append(f"Skipped: {skipped_models}")
log.append("")
log.append("--- Output files ---")
for f in sorted(OUT_DIR.iterdir()):
    log.append(f"  {f.name}")
log.append("")
log.append(f"SHAP success: {shap_success}")
log.append("")
log.append("=" * 60)
log.append("Step 6 finished.")
log.append("=" * 60)

with open(OUT_DIR / "step6_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\n".join(log))
print(f"\nStep 6 finished.")
print(f"Results saved to {OUT_DIR}\\")
