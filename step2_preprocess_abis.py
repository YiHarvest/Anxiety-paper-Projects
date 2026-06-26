"""
Step 2: Blood biomarker preprocessing, descriptive statistics, and ABIS index calculation.
Anxiety binary classification experiment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import pickle
import logging

# ============================================================
# Path configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "dataset" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step2_preprocess_abis"

TRAIN_PATH = INPUT_DIR / "anxiety_bio15_seed284_train_sorted_by_difficulty.csv"
TEST_PATH = INPUT_DIR / "anxiety_bio15_seed284_test_sorted_by_difficulty.csv"

# ============================================================
# Column definitions
# ============================================================
TARGET = "Anxiety_14"

# Columns forbidden as model features
FORBIDDEN_FEATURES = [
    "CaseNumber",
    "Depression_18",
    "Chronic_pain",
]

# Auxiliary columns (also not model features)
AUXILIARY_COLUMNS = [
    "difficulty_rank_in_split_easy_to_hard",
    "split",
    "split_seed",
    "stratified_split",
    "model_feature_set",
    "original_index_0based",
    "original_csv_row_1based_including_header",
    "predicted_probability_anxiety_1",
    "predicted_label_threshold_0_5",
    "is_correct_at_threshold_0_5",
    "true_class_probability",
    "difficulty_score_0_easy_1_hard",
    "discrimination_confidence_0_uncertain_1_confident",
    "difficulty_level",
    "sorting_remark_why_this_rank",
]

# 6 raw blood biomarkers
RAW_BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]

# 9 ratio biomarkers
RATIO_BIOMARKERS = [
    "IL6/IL10",
    "TNFalpha/IL10",
    "CRP/IL10",
    "CORT/ACTH",
    "CORT/IL6",
    "CORT/CRP",
    "IL6/TNFalpha",
    "CRP/IL6",
    "ACTH/IL6",
]

# All 15 blood features
ALL_BLOOD_FEATURES = RAW_BIOMARKERS + RATIO_BIOMARKERS

# Columns to keep in model version
MODEL_COLUMNS = [TARGET] + ALL_BLOOD_FEATURES

# ============================================================
# Setup logging
# ============================================================
LOG_PATH = OUTPUT_DIR / "step2_log.txt"

# We'll set up the logger after creating the output dir, but collect log lines in a list first.
log_lines = []

def log(msg: str):
    """Record a log message (will be written to file at the end)."""
    log_lines.append(msg)
    print(msg)


# ============================================================
# Helper: check required columns exist
# ============================================================
def check_columns(df: pd.DataFrame, required_cols: list, df_name: str):
    """Check that all required columns exist in the dataframe. Raise if not."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        err = f"ERROR: Missing columns in {df_name}: {missing}"
        log(err)
        raise KeyError(err)


# ============================================================
def main():
    # ---- 1. Read data ----
    log(f"Input train path: {TRAIN_PATH}")
    log(f"Input test path: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    log(f"Training set N: {len(train)}")
    log(f"Test set N: {len(test)}")

    # Verify required columns exist
    required_cols = [TARGET] + RAW_BIOMARKERS + RATIO_BIOMARKERS
    check_columns(train, required_cols, "train")
    check_columns(test, required_cols, "test")

    # ---- 2. Create output directory ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {OUTPUT_DIR}")

    # ---- 3. Table 1: Split distribution check ----
    def make_dist_table(df: pd.DataFrame, label: str) -> dict:
        n = len(df)
        n0 = int((df[TARGET] == 0).sum())
        n1 = int((df[TARGET] == 1).sum())
        ratio = n1 / n if n > 0 else 0
        return {
            "Dataset": label,
            "N": n,
            "Anxiety_0": n0,
            "Anxiety_1": n1,
            "Anxiety_1_ratio": round(ratio, 4),
        }

    table1 = pd.DataFrame([
        make_dist_table(train, "train"),
        make_dist_table(test, "test"),
    ])
    table1_path = OUTPUT_DIR / "table1_split_distribution_check.csv"
    table1.to_csv(table1_path, index=False)
    log(f"Saved: {table1_path}")
    log(f"Training set Anxiety_14 distribution: {dict(table1.iloc[0])}")
    log(f"Test set Anxiety_14 distribution: {dict(table1.iloc[1])}")

    # ---- 4. Table 2: Biomarker descriptive statistics ----
    log(f"Raw biomarker names: {RAW_BIOMARKERS}")
    log(f"Ratio biomarker names: {RATIO_BIOMARKERS}")

    desc_rows = []
    for var in RAW_BIOMARKERS:
        g0 = train.loc[train[TARGET] == 0, var].dropna()
        g1 = train.loc[train[TARGET] == 1, var].dropna()
        try:
            stat, p = mannwhitneyu(g0, g1, alternative="two-sided")
        except ValueError:
            p = np.nan
        desc_rows.append({
            "variable": var,
            "group0_median": g0.median(),
            "group0_Q1": g0.quantile(0.25),
            "group0_Q3": g0.quantile(0.75),
            "group1_median": g1.median(),
            "group1_Q1": g1.quantile(0.25),
            "group1_Q3": g1.quantile(0.75),
            "p_value": p,
        })

    table2 = pd.DataFrame(desc_rows)
    table2_path = OUTPUT_DIR / "table2_biomarker_description.csv"
    table2.to_csv(table2_path, index=False)
    log(f"Saved: {table2_path}")

    # ---- 5. Preprocessing pipeline ----
    # Log missing value counts per biomarker
    log("Missing value counts per biomarker (train):")
    for var in ALL_BLOOD_FEATURES:
        n_miss = int(train[var].isna().sum())
        log(f"  {var}: {n_miss}")

    # Step 1: Median imputation (from train)
    train_medians = {}
    for var in ALL_BLOOD_FEATURES:
        train_medians[var] = train[var].median()

    log("Training set medians for imputation:")
    for var, med in train_medians.items():
        log(f"  {var}: {med}")

    train_imputed = train.copy()
    test_imputed = test.copy()
    for var in ALL_BLOOD_FEATURES:
        train_imputed[var] = train_imputed[var].fillna(train_medians[var])
        test_imputed[var] = test_imputed[var].fillna(train_medians[var])

    # Step 2: log1p transformation
    for var in ALL_BLOOD_FEATURES:
        train_imputed[var] = np.log1p(train_imputed[var])
        test_imputed[var] = np.log1p(test_imputed[var])

    # Step 3: Winsorization (1%, 99% thresholds from train)
    winsor_low = {}
    winsor_high = {}
    for var in ALL_BLOOD_FEATURES:
        lo = train_imputed[var].quantile(0.01)
        hi = train_imputed[var].quantile(0.99)
        winsor_low[var] = lo
        winsor_high[var] = hi

    log("Winsorization thresholds (1% / 99% from train log1p):")
    for var in ALL_BLOOD_FEATURES:
        log(f"  {var}: lower={winsor_low[var]:.6f}, upper={winsor_high[var]:.6f}")

    for var in ALL_BLOOD_FEATURES:
        train_imputed[var] = train_imputed[var].clip(winsor_low[var], winsor_high[var])
        test_imputed[var] = test_imputed[var].clip(winsor_low[var], winsor_high[var])

    # Step 4: Z-score standardization (params from train)
    train_mean = {}
    train_std = {}
    for var in ALL_BLOOD_FEATURES:
        train_mean[var] = train_imputed[var].mean()
        train_std[var] = train_imputed[var].std()

    log("Standardization parameters (mean / std from train after winsor):")
    for var in ALL_BLOOD_FEATURES:
        log(f"  {var}: mean={train_mean[var]:.6f}, std={train_std[var]:.6f}")

    train_preprocessed = train_imputed.copy()
    test_preprocessed = test_imputed.copy()
    for var in ALL_BLOOD_FEATURES:
        train_preprocessed[var] = (train_preprocessed[var] - train_mean[var]) / train_std[var]
        test_preprocessed[var] = (test_preprocessed[var] - train_mean[var]) / train_std[var]

    # ---- 6. Save preprocessed data ----
    # Full versions
    train_full_path = OUTPUT_DIR / "train_preprocessed_full.csv"
    test_full_path = OUTPUT_DIR / "test_preprocessed_full.csv"
    train_preprocessed.to_csv(train_full_path, index=False)
    test_preprocessed.to_csv(test_full_path, index=False)
    log(f"Saved: {train_full_path}")
    log(f"Saved: {test_full_path}")

    # Model versions
    train_model_path = OUTPUT_DIR / "train_preprocessed_model.csv"
    test_model_path = OUTPUT_DIR / "test_preprocessed_model.csv"
    train_preprocessed[MODEL_COLUMNS].to_csv(train_model_path, index=False)
    test_preprocessed[MODEL_COLUMNS].to_csv(test_model_path, index=False)
    log(f"Saved: {train_model_path}")
    log(f"Saved: {test_model_path}")

    # ---- 7. Compute ABIS index ----
    # ABIS = mean[z(log IL6), z(log TNFalpha), z(log CRP)] - z(log IL10) + z(log CORT) - z(log ACTH)
    # Using the already-standardized raw biomarkers from step 5.
    pro_inflammatory_mean = (
        train_preprocessed["IL6"] + train_preprocessed["TNFalpha"] + train_preprocessed["CRP"]
    ) / 3.0
    abis_train = pro_inflammatory_mean - train_preprocessed["IL10"] + train_preprocessed["CORT"] - train_preprocessed["ACTH"]

    pro_inflammatory_mean_test = (
        test_preprocessed["IL6"] + test_preprocessed["TNFalpha"] + test_preprocessed["CRP"]
    ) / 3.0
    abis_test = pro_inflammatory_mean_test - test_preprocessed["IL10"] + test_preprocessed["CORT"] - test_preprocessed["ACTH"]

    log(f"ABIS training set: mean={abis_train.mean():.6f}, std={abis_train.std():.6f}")
    log(f"ABIS test set: mean={abis_test.mean():.6f}, std={abis_test.std():.6f}")

    # ---- 8. Save data with ABIS ----
    train_abis = train_preprocessed.copy()
    train_abis["ABIS"] = abis_train
    test_abis = test_preprocessed.copy()
    test_abis["ABIS"] = abis_test

    # Full versions
    train_abis_full_path = OUTPUT_DIR / "train_with_abis_full.csv"
    test_abis_full_path = OUTPUT_DIR / "test_with_abis_full.csv"
    train_abis.to_csv(train_abis_full_path, index=False)
    test_abis.to_csv(test_abis_full_path, index=False)
    log(f"Saved: {train_abis_full_path}")
    log(f"Saved: {test_abis_full_path}")

    # Model versions
    abis_model_cols = MODEL_COLUMNS + ["ABIS"]
    train_abis_model_path = OUTPUT_DIR / "train_with_abis_model.csv"
    test_abis_model_path = OUTPUT_DIR / "test_with_abis_model.csv"
    train_abis[abis_model_cols].to_csv(train_abis_model_path, index=False)
    test_abis[abis_model_cols].to_csv(test_abis_model_path, index=False)
    log(f"Saved: {train_abis_model_path}")
    log(f"Saved: {test_abis_model_path}")

    # ---- 9. Save preprocessing parameters ----
    preprocess_params = {
        "train_medians": train_medians,
        "winsor_low_log1p": winsor_low,
        "winsor_high_log1p": winsor_high,
        "train_mean_after_winsor": train_mean,
        "train_std_after_winsor": train_std,
    }
    params_path = OUTPUT_DIR / "preprocess_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(preprocess_params, f)
    log(f"Saved: {params_path}")

    # ---- 11. Write log file (before plot to ensure it's saved) ----
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Log saved to: {LOG_PATH}")

    # ---- 10. ABIS distribution plot ----
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))

    g0_abis = train_abis.loc[train_abis[TARGET] == 0, "ABIS"]
    g1_abis = train_abis.loc[train_abis[TARGET] == 1, "ABIS"]

    box_data = [g0_abis.values, g1_abis.values]
    bp = ax.boxplot(
        box_data,
        tick_labels=["Anxiety=0", "Anxiety=1"],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
    )

    # Box colors
    bp["boxes"][0].set_facecolor("#4ECDC4")
    bp["boxes"][1].set_facecolor("#FF6B6B")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Add jittered scatter points
    for i, data_vals in enumerate(box_data):
        jitter = np.random.normal(i + 1, 0.04, size=len(data_vals))
        non_nan = ~np.isnan(data_vals)
        ax.scatter(
            jitter[non_nan], np.array(data_vals)[non_nan],
            alpha=0.3, s=10, color="black", zorder=3,
        )

    ax.set_ylabel("ABIS", fontfamily="Times New Roman")
    ax.set_title("ABIS Distribution by Anxiety Status (Training Set)", fontfamily="Times New Roman")
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()

    png_path = OUTPUT_DIR / "figure1_abis_distribution.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log(f"Saved: {png_path}")

    # ---- 12. Done ----
    print()
    print("Step 2 finished.")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
