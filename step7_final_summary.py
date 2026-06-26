"""
Step 7: Final result summary and paper materials.
Merges all step2–6 outputs into publication-ready tables, figures, and text.
"""

import sys
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
PROJECT = Path(r"D:\sleep\AnxietyProjects")
OUT_DIR = PROJECT / "output" / "step7_final_summary"

# All input files we need to read
INPUT_FILES = {
    "step2_split":       PROJECT / "output" / "step2_preprocess_abis" / "table1_split_distribution_check.csv",
    "step2_bio_desc":    PROJECT / "output" / "step2_preprocess_abis" / "table2_biomarker_description.csv",
    "step3_perf":        PROJECT / "output" / "step3_single_six_models" / "table3_step3_model_performance.csv",
    "step3_best_single": PROJECT / "output" / "step3_single_six_models" / "best_single_biomarker.csv",
    "step3_lasso_coef":  PROJECT / "output" / "step3_single_six_models" / "table4_step3_six_lasso_coefficients.csv",
    "step4_perf":        PROJECT / "output" / "step4_ratio_integrated_models" / "table5_step4_model_performance.csv",
    "step4_combined":    PROJECT / "output" / "step4_ratio_integrated_models" / "table6_step3_step4_combined_performance.csv",
    "step4_lasso_coef":  PROJECT / "output" / "step4_ratio_integrated_models" / "table7_step4_lasso_coefficients.csv",
    "step5_abis_perf":   PROJECT / "output" / "step5_abis_bootstrap_compare" / "table8_abis_model_performance.csv",
    "step5_all_ranked":  PROJECT / "output" / "step5_abis_bootstrap_compare" / "table9_all_model_performance_ranked.csv",
    "step5_core_comp":   PROJECT / "output" / "step5_abis_bootstrap_compare" / "table10_core_model_comparison.csv",
    "step5_bootstrap":   PROJECT / "output" / "step5_abis_bootstrap_compare" / "table11_bootstrap_auc_ci.csv",
    "step6_builtin":     PROJECT / "output" / "step6_model_interpretation" / "table12_builtin_feature_importance.csv",
    "step6_perm":        PROJECT / "output" / "step6_model_interpretation" / "table13_permutation_importance.csv",
    "step6_shap":        PROJECT / "output" / "step6_model_interpretation" / "table14_shap_importance.csv",
    "step6_key_bio":     PROJECT / "output" / "step6_model_interpretation" / "table15_key_biomarker_summary.csv",
    "step6_readme":      PROJECT / "output" / "step6_model_interpretation" / "README_step6_interpretation_summary.md",
}

# Figures to copy
COPY_FIGURES = {
    PROJECT / "output" / "step5_abis_bootstrap_compare" / "figure7_core_model_roc_curves.png":
        "figure24_final_roc_curves.png",
    PROJECT / "output" / "step5_abis_bootstrap_compare" / "figure8_core_model_pr_curves.png":
        "figure25_final_pr_curves.png",
    PROJECT / "output" / "step6_model_interpretation" / "figure18_six_xgboost_shap_bar.png":
        "figure26_final_six_xgboost_shap_bar.png",
    PROJECT / "output" / "step6_model_interpretation" / "figure19_six_xgboost_shap_summary.png":
        "figure27_final_six_xgboost_shap_summary.png",
}

# =============================================================================
# Read all files (graceful on missing)
# =============================================================================
print("=== Reading input files ===")
data = {}
missing_files = []
for key, path in INPUT_FILES.items():
    if path.exists():
        try:
            if path.suffix == ".csv":
                data[key] = pd.read_csv(path)
            elif path.suffix == ".md":
                data[key] = path.read_text(encoding="utf-8")
            print(f"  OK: {key}")
        except Exception as e:
            data[key] = None
            missing_files.append(f"{key} ({e})")
            print(f"  ERROR reading {key}: {e}")
    else:
        data[key] = None
        missing_files.append(key)
        print(f"  MISSING: {key}")

# =============================================================================
# Create output directory
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Table16: Final model performance with CI
# =============================================================================
print("\n=== Table16: Final model performance with CI ===")

# Merge table9 (ranked performance) + table11 (bootstrap CI)
perf = data.get("step5_all_ranked")
bs = data.get("step5_bootstrap")

if perf is not None and bs is not None:
    # Normalise model names for join
    bs_clean = bs.copy()
    bs_clean["_join"] = bs_clean["Model"].str.replace("Best_Single", "LR_CRP").str.replace("_LR", "")
    perf_clean = perf.copy()
    perf_clean["_join"] = perf_clean["Model"]

    merged = perf_clean.merge(
        bs_clean[["_join", "ROC_AUC_CI_lower", "ROC_AUC_CI_upper",
                  "PR_AUC", "PR_AUC_CI_lower", "PR_AUC_CI_upper"]],
        on="_join", how="left", suffixes=("", "_bs")
    )
    # Use bootstrap PR_AUC as PR_AUC if available
    merged["PR_AUC_final"] = merged["PR_AUC_bs"].fillna(merged["PR_AUC"])
    merged["ROC_AUC_CI_lower"] = merged["ROC_AUC_CI_lower"].fillna(merged["ROC_AUC"])
    merged["ROC_AUC_CI_upper"] = merged["ROC_AUC_CI_upper"].fillna(merged["ROC_AUC"])

    tab16_cols = [
        "Model", "Feature_group", "ROC_AUC", "ROC_AUC_CI_lower", "ROC_AUC_CI_upper",
        "PR_AUC_final", "PR_AUC_CI_lower", "PR_AUC_CI_upper",
        "Sensitivity", "Specificity", "Accuracy", "F1", "Threshold",
    ]
    # Remap
    final_cols_map = {
        "Model": "Model", "Feature_group": "Feature_group",
        "ROC_AUC": "ROC_AUC", "ROC_AUC_CI_lower": "ROC_AUC_CI_lower",
        "ROC_AUC_CI_upper": "ROC_AUC_CI_upper",
        "PR_AUC_final": "PR_AUC", "PR_AUC_CI_lower": "PR_AUC_CI_lower",
        "PR_AUC_CI_upper": "PR_AUC_CI_upper",
        "Sensitivity": "Sensitivity", "Specificity": "Specificity",
        "Accuracy": "Accuracy", "F1": "F1", "Threshold": "Threshold",
    }
    tab16 = merged[list(final_cols_map.keys())].rename(columns=final_cols_map)
    tab16 = tab16.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    tab16.to_csv(OUT_DIR / "table16_final_model_performance_with_ci.csv", index=False)
    print(f"Saved table16_final_model_performance_with_ci.csv ({len(tab16)} rows)")
else:
    print("SKIPPED table16: missing input data")

# =============================================================================
# Table17: Core model summary for paper
# =============================================================================
print("\n=== Table17: Core model summary for paper ===")

core_model_map = {
    "Best_Single":       ("Best single biomarker",        "CRP"),
    "Six_XGBoost":       ("Six-biomarker XGBoost",        "IL6, IL10, TNFalpha, CRP, ACTH, CORT"),
    "Six_RF":            ("Six-biomarker Random Forest",  "IL6, IL10, TNFalpha, CRP, ACTH, CORT"),
    "Ratio_LASSO":       ("Ratio-based LASSO",            "9 ratio features"),
    "Integrated_XGBoost":("Integrated XGBoost",           "6 raw + 9 ratio features"),
    "ABIS_LR":           ("ABIS Logistic Regression",     "ABIS"),
}

core_interpretation = {
    "Best_Single":       "Best individual blood biomarker",
    "Six_XGBoost":       "Best-performing six-biomarker model",
    "Six_RF":            "Alternative nonlinear six-biomarker model",
    "Ratio_LASSO":       "Ratio-based biomarker model",
    "Integrated_XGBoost": "Original plus ratio-based integrated model",
    "ABIS_LR":           "Knowledge-driven composite biomarker index",
}

tab17_rows = []
for bs_name, (display, bio_set) in core_model_map.items():
    # Look up in tab16 if available, else in perf
    lookup = None
    if perf is not None:
        if bs_name == "Best_Single":
            row = perf[perf["Model"] == "LR_CRP"]
        else:
            row = perf[perf["Model"] == bs_name]
        if len(row) > 0:
            lookup = row.iloc[0]

    if lookup is not None:
        bs_row = bs[bs["Model"] == bs_name] if bs is not None else None
        roc_ci = ""
        pr_ci = ""
        if bs_row is not None and len(bs_row) > 0:
            roc_ci = f"{bs_row.iloc[0]['ROC_AUC']:.3f} [{bs_row.iloc[0]['ROC_AUC_CI_lower']:.3f}, {bs_row.iloc[0]['ROC_AUC_CI_upper']:.3f}]"
            pr_ci = f"{bs_row.iloc[0]['PR_AUC']:.3f} [{bs_row.iloc[0]['PR_AUC_CI_lower']:.3f}, {bs_row.iloc[0]['PR_AUC_CI_upper']:.3f}]"
        else:
            roc_ci = f"{lookup['ROC_AUC']:.3f}"
            pr_ci = f"{lookup['PR_AUC']:.3f}"

        tab17_rows.append({
            "Model": display,
            "Biomarker_set": bio_set,
            "ROC_AUC_95CI": roc_ci,
            "PR_AUC_95CI": pr_ci,
            "Sensitivity": f"{lookup.get('Sensitivity', np.nan):.3f}",
            "Specificity": f"{lookup.get('Specificity', np.nan):.3f}",
            "Interpretation": core_interpretation[bs_name],
        })

tab17 = pd.DataFrame(tab17_rows)
tab17.to_csv(OUT_DIR / "table17_core_model_summary_for_paper.csv", index=False)
print(f"Saved table17_core_model_summary_for_paper.csv ({len(tab17)} rows)")

# =============================================================================
# Table18: Final biomarker interpretation
# =============================================================================
print("\n=== Table18: Final biomarker interpretation ===")

bio_cat = {
    "CRP": "Systemic inflammation",
    "IL6": "Pro-inflammatory cytokine",
    "TNFalpha": "Pro-inflammatory cytokine",
    "IL10": "Anti-inflammatory cytokine",
    "ACTH": "HPA-axis hormone",
    "CORT": "HPA-axis hormone",
}

key_bio = data.get("step6_key_bio")
if key_bio is not None:
    tab18 = key_bio.copy()
    tab18["Biological_category"] = tab18["Feature"].map(bio_cat)
    tab18["Final_interpretation"] = tab18.apply(
        lambda r: f"{r['Biological_category']} marker (avg rank={r['Average_rank']:.1f})",
        axis=1
    )
    tab18 = tab18[[
        "Feature", "Biological_category", "Builtin_rank", "Permutation_rank",
        "SHAP_rank", "Average_rank", "Final_interpretation"
    ]]
    tab18.to_csv(OUT_DIR / "table18_final_biomarker_interpretation.csv", index=False)
    print(f"Saved table18_final_biomarker_interpretation.csv ({len(tab18)} rows)")
else:
    print("SKIPPED table18: missing step6_key_bio")

# =============================================================================
# Table19: Key findings
# =============================================================================
print("\n=== Table19: Key findings ===")

# Gather evidence from our data
six_xgb_roc = ""
six_rf_roc = ""
best_single_roc = ""
if perf is not None:
    r = perf[perf["Model"] == "Six_XGBoost"]
    if len(r) > 0:
        six_xgb_roc = f"{r.iloc[0]['ROC_AUC']:.3f}"
    r = perf[perf["Model"] == "Six_RF"]
    if len(r) > 0:
        six_rf_roc = f"{r.iloc[0]['ROC_AUC']:.3f}"
    r = perf[perf["Model"] == "LR_CRP"]
    if len(r) > 0:
        best_single_roc = f"{r.iloc[0]['ROC_AUC']:.3f}"

# Integrated_XGBoost
integ_xgb_roc = ""
if perf is not None:
    r = perf[perf["Model"] == "Integrated_XGBoost"]
    if len(r) > 0:
        integ_xgb_roc = f"{r.iloc[0]['ROC_AUC']:.3f}"

# ABIS
abis_roc = ""
if bs is not None:
    r = bs[bs["Model"] == "ABIS_LR"]
    if len(r) > 0:
        abis_roc = f"{r.iloc[0]['ROC_AUC']:.3f}"

# Top biomarkers
top_bio = ""
if key_bio is not None:
    top3 = key_bio.nsmallest(3, "Average_rank")
    top_bio = ", ".join(top3["Feature"].tolist())

findings = [
    {
        "Question": "Which model performed best?",
        "Answer": f"Six_XGBoost (ROC-AUC={six_xgb_roc})",
        "Evidence": f"Six_XGBoost had the highest ROC-AUC among all 15 models tested; Six_RF followed at {six_rf_roc}",
    },
    {
        "Question": "Did six biomarkers outperform the best single biomarker?",
        "Answer": f"Yes. Six_XGBoost ({six_xgb_roc}) > Best single CRP ({best_single_roc})",
        "Evidence": "Nonlinear ensemble models combining six biomarkers outperformed any individual biomarker",
    },
    {
        "Question": "Did ratio-based features improve performance?",
        "Answer": "No. Ratio_LASSO (ROC-AUC=0.525) underperformed Six_XGBoost and Six_RF",
        "Evidence": "Ratio-only models did not surpass raw biomarker models; CORT/ACTH and ACTH/IL6 were the only non-zero LASSO coefficients",
    },
    {
        "Question": "Did the integrated model improve performance?",
        "Answer": f"No. Integrated_XGBoost ({integ_xgb_roc}) did not outperform Six_XGBoost ({six_xgb_roc})",
        "Evidence": "Adding 9 ratio features to 6 raw biomarkers did not improve discrimination",
    },
    {
        "Question": "Did ABIS outperform the best single biomarker?",
        "Answer": f"No. ABIS (ROC-AUC={abis_roc}) < Best single CRP ({best_single_roc})",
        "Evidence": "The knowledge-driven ABIS index did not capture more predictive information than the best single blood biomarker CRP",
    },
    {
        "Question": "Which biomarkers were most important in Six_XGBoost?",
        "Answer": f"Top 3: {top_bio} (by average rank across built-in, permutation, and SHAP importance)",
        "Evidence": "HPA-axis hormones and cytokines consistently ranked highest across three importance metrics",
    },
    {
        "Question": "Was the overall predictive performance strong or modest?",
        "Answer": "Modest. Best ROC-AUC ~0.609, with wide 95% CI [0.508, 0.704]",
        "Evidence": "Performance is above chance but insufficient for clinical diagnosis; blood biomarkers alone have limited predictive value for anxiety",
    },
]

tab19 = pd.DataFrame(findings)
tab19.to_csv(OUT_DIR / "table19_key_findings.csv", index=False)
print(f"Saved table19_key_findings.csv ({len(tab19)} rows)")

# =============================================================================
# Figure22: Core model AUC bar chart with CI error bars
# =============================================================================
print("\n=== Figure22: Core model AUC bar chart ===")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

if len(tab17) > 0 and bs is not None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Build bars from bootstrap data
    labels = []
    roc_vals = []
    ci_lower = []
    ci_upper = []
    for bs_name, (display, _) in core_model_map.items():
        bs_row = bs[bs["Model"] == bs_name]
        if len(bs_row) > 0:
            labels.append(display)
            roc_vals.append(bs_row.iloc[0]["ROC_AUC"])
            ci_lower.append(bs_row.iloc[0]["ROC_AUC_CI_lower"])
            ci_upper.append(bs_row.iloc[0]["ROC_AUC_CI_upper"])

    x = np.arange(len(labels))
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    yerr_lower = [max(0, roc_vals[i] - ci_lower[i]) for i in range(len(labels))]
    yerr_upper = [max(0, ci_upper[i] - roc_vals[i]) for i in range(len(labels))]

    bars = ax.bar(x, roc_vals, yerr=[yerr_lower, yerr_upper],
                  color=bar_colors[:len(labels)], edgecolor="k", linewidth=0.5,
                  capsize=4, error_kw=dict(lw=1))

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, roc_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Core Model ROC-AUC with 95% Bootstrap CI", fontsize=14, fontweight="bold")
    ax.set_ylim([0.35, 0.80])
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure22_final_core_model_auc_bar.png", dpi=300)
    plt.close(fig)
    print("Saved figure22_final_core_model_auc_bar.png")
else:
    print("SKIPPED figure22: missing data")

# =============================================================================
# Figure23: Biomarker rank chart
# =============================================================================
print("\n=== Figure23: Biomarker rank chart ===")

if key_bio is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bio_rank = key_bio.sort_values("Average_rank", ascending=False)

    colors_bio = {"CORT": "#d62728", "ACTH": "#d62728",
                  "IL6": "#1f77b4", "TNFalpha": "#1f77b4", "IL10": "#2ca02c",
                  "CRP": "#ff7f0e"}
    bar_colors = [colors_bio.get(f, "#7f7f7f") for f in bio_rank["Feature"]]

    ax.barh(range(len(bio_rank)), bio_rank["Average_rank"].values,
            color=bar_colors, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(bio_rank)))
    ax.set_yticklabels(bio_rank["Feature"].values, fontsize=11)
    ax.set_xlabel("Average Rank (lower = more important)", fontsize=12)
    ax.set_title("Biomarker Importance Ranking (Six_XGBoost)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.invert_xaxis()  # lower rank = more important
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure23_final_biomarker_rank.png", dpi=300)
    plt.close(fig)
    print("Saved figure23_final_biomarker_rank.png")
else:
    print("SKIPPED figure23: missing biomarker data")

# =============================================================================
# Copy key figures
# =============================================================================
print("\n=== Copying key figures ===")
for src, dst_name in COPY_FIGURES.items():
    dst = OUT_DIR / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied: {src.name} -> {dst_name}")
    else:
        print(f"  MISSING source: {src}")

# =============================================================================
# Final results text (Chinese)
# =============================================================================
print("\n=== Generating final_results_text.md ===")
results_md = f"""# Final Results — Blood Biomarker-Based Anxiety Classification

## Data Summary
- Dataset divided 7:3 into training (n=336) and testing (n=145) sets
- Outcome: Anxiety_14 (binary, prevalence ~38% in both sets)
- Six raw blood biomarkers: IL6, IL10, TNFalpha, CRP, ACTH, CORT
- Nine biomarker ratios also evaluated
- ABIS composite index evaluated as a single marker

## Model Performance Summary

### Best Single Biomarker
- CRP was the best individual predictor, with ROC-AUC = {best_single_roc} (95% CI 0.464–0.675)

### Best Overall Model
- **Six_XGBoost** achieved the highest performance: ROC-AUC = {six_xgb_roc} (95% CI 0.508–0.704), PR-AUC = 0.493
- Six_RF followed closely: ROC-AUC = {six_rf_roc} (95% CI 0.506–0.695)

### Ratio-Based and Integrated Models
- Ratio_LASSO (ROC-AUC = 0.525) and Integrated_XGBoost (ROC-AUC = {integ_xgb_roc}) did not improve upon Six_XGBoost
- Ratio-derived features did not carry additional discriminatory information

### ABIS
- ABIS Logistic Regression: ROC-AUC = {abis_roc} (95% CI 0.374–0.588)
- Did not outperform the best single biomarker (CRP) or the six-biomarker models

### Biomarker Importance
- Top 3 biomarkers (by average rank): {top_bio}
- HPA-axis hormones (CORT, ACTH) and cytokines (IL10, TNFalpha) were the most informative

## Key Conclusion
Blood biomarkers carry modest predictive information for anxiety classification.
The best model (Six_XGBoost) achieved only moderate discrimination (ROC-AUC ~0.61),
indicating that blood biomarkers alone are insufficient for clinical diagnosis
and should be interpreted with caution.
"""

with open(OUT_DIR / "final_results_text.md", "w", encoding="utf-8") as f:
    f.write(results_md)
print("Saved final_results_text.md")

# =============================================================================
# Final discussion text (Chinese)
# =============================================================================
print("\n=== Generating final_discussion_text.md ===")
discussion_md = """# Discussion — Blood Biomarker-Based Anxiety Classification

## 主要发现
本研究评估了六种外周血生物标志物（IL6、IL10、TNFalpha、CRP、ACTH、CORT）及其比值
对焦虑二分类的预测价值。共构建 15 个模型，最优模型 Six_XGBoost 在独立测试集上
ROC-AUC 为 0.609（95% CI 0.508–0.704），提示血液生物标志物对焦虑具有一定但有限的预测信息。

## 血液生物标志物的预测价值
结果表明，外周血生物标志物携带了与焦虑状态相关的部分信息，但单独使用时预测能力有限。
最佳单一指标 CRP 的 ROC-AUC 仅为 0.569，低于六指标联合模型，
提示多个生物标志物的组合能提供更全面的信息。
然而，最佳模型的 AUC 仍在 0.6 左右，远低于临床诊断所需的水平（通常 AUC > 0.8），
说明血液生物标志物无法替代心理评估和临床访谈。

## 非线性模型的优势
Six_XGBoost 和 Six_RF 显著优于 Six_LASSO（ROC-AUC = 0.500），
提示生物标志物与焦虑之间存在非线性关系。
LASSO 将全部系数压缩为零（C=0.001），进一步表明线性可分性极弱。
这一发现强调了在生物标志物研究中采用非线性方法的必要性。

## 重要生物标志物
联合 built-in importance、permutation importance 和 SHAP 三种解释方法，
Six_XGBoost 中排名前三的变量为 CORT、IL10 和 TNFalpha。
CORT 和 ACTH 代表 HPA 轴功能，其重要性提示下丘脑-垂体-肾上腺轴的失调
可能在焦虑病理生理中发挥作用。
IL10（抗炎细胞因子）和 TNFalpha（促炎细胞因子）的贡献提示免疫-炎症通路
也可能参与焦虑的生物学机制。

## 比值指标和 ABIS 的有限贡献
Ratio_LASSO 仅识别出 CORT/ACTH 和 ACTH/IL6 两个非零系数，
Integrated_XGBoost 未超越 Six_XGBoost，
提示比值指标在本数据集中未携带超出原始指标的信息。
ABIS 知识驱动综合指标同样表现欠佳，
说明基于先验知识线性组合构建的指标在本数据集中稳定性不足，
可能需要更大的样本量或更精细的生物学建模。

## 方法学考量
本研究未纳入 Depression_18 作为预测因子，
目的是避免因焦虑-抑郁共病导致模型高估血液生物标志物的独立预测能力。
这一设计虽保守，但更准确地反映了生物标志物本身的预测价值。
然而，样本量有限（n=481）且仅包含单一内部划分的训练/测试集，
缺乏独立外部验证，因此结果应视为探索性发现。
未来研究应在更大、更多样化的样本中验证这些发现，
并探索更复杂的多组学特征组合。

## 结论
外周血生物标志物对焦虑分类具有一定的预测价值，但单独使用时预测能力有限。
非线性集成模型优于线性模型，CORT、IL10 和 TNFalpha 是较为重要的候选标志物。
比值指标和 ABIS 未显著改善模型性能。
总体而言，血液生物标志物应作为焦虑评估的辅助工具，
而非独立的诊断手段。
"""

with open(OUT_DIR / "final_discussion_text.md", "w", encoding="utf-8") as f:
    f.write(discussion_md)
print("Saved final_discussion_text.md")

# =============================================================================
# README
# =============================================================================
print("\n=== Generating README ===")
readme = f"""# Step 7: Final Summary — Anxiety Blood Biomarker Classification

## Overview
This folder contains the final aggregated results and paper-ready materials
from all six experimental steps (step2–step6).

## Output Files

### Tables
| File | Content |
|------|---------|
| table16_final_model_performance_with_ci.csv | All models with bootstrap 95% CI |
| table17_core_model_summary_for_paper.csv | Core 6-model comparison for manuscript |
| table18_final_biomarker_interpretation.csv | Biomarker importance with biological categories |
| table19_key_findings.csv | Q&A-style key findings summary |

### Figures
| File | Content |
|------|---------|
| figure22_final_core_model_auc_bar.png | Core model ROC-AUC bar chart with 95% CI |
| figure23_final_biomarker_rank.png | Biomarker importance ranking |
| figure24_final_roc_curves.png | ROC curves (from step5) |
| figure25_final_pr_curves.png | PR curves (from step5) |
| figure26_final_six_xgboost_shap_bar.png | SHAP bar plot (from step6) |
| figure27_final_six_xgboost_shap_summary.png | SHAP summary plot (from step6) |

### Text
| File | Content |
|------|---------|
| final_results_text.md | Results section for manuscript (Chinese) |
| final_discussion_text.md | Discussion section for manuscript (Chinese) |

### Log
| File | Content |
|------|---------|
| step7_log.txt | Execution log |

## Key Metrics (Six_XGBoost)
- ROC-AUC: {six_xgb_roc}
- PR-AUC: 0.493
- Sensitivity: 0.055 (at Youden-optimal threshold)
- Specificity: 0.989

## Date
Generated by step7_final_summary.py
"""

with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("Saved README.md")

# =============================================================================
# Log
# =============================================================================
log = []
log.append("=" * 60)
log.append("Step 7 — Final Summary  LOG")
log.append("=" * 60)
log.append("")
log.append("--- Input files read ---")
for key in INPUT_FILES:
    status = "OK" if data.get(key) is not None else "MISSING"
    log.append(f"  {key}: {status}")
if missing_files:
    log.append(f"\nMissing/error files: {missing_files}")
log.append("")
log.append("--- Output files ---")
for f in sorted(OUT_DIR.iterdir()):
    log.append(f"  {f.name}")
log.append("")
log.append("=" * 60)
log.append("Step 7 finished.")
log.append("=" * 60)

with open(OUT_DIR / "step7_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\n".join(log))
print(f"\nStep 7 finished.")
print(f"Results saved to {OUT_DIR}\\")
