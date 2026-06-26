"""
Step 3 + 高斯噪声增强：对比原始数据与增强数据的模型性能。

基于 step3_single_six_models.py，在训练集上应用高斯噪声增强，
比较增强前后模型性能变化。

增强策略：
1. 基础高斯噪声：统一噪声强度
2. 特征自适应噪声：不同生物标志物使用不同噪声强度
3. 可选：仅增强少数类（焦虑阳性）

输出：
- 增强后的训练数据
- 性能对比表格
- ROC/PR 曲线对比图
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

# Try importing xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# =============================================================================
# 路径配置
# =============================================================================
PROJECT = Path(r"D:\sleep\AnxietyProjects")
TRAIN_PATH = PROJECT / "output" / "step2_preprocess_abis" / "train_with_abis_model.csv"
TEST_PATH = PROJECT / "output" / "step2_preprocess_abis" / "test_with_abis_model.csv"
OUT_DIR = PROJECT / "output" / "step3_gaussian_augmentation"

# 六项原始血液生物标志物
BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]

# 排除列
EXCLUDE_COLS = [
    "CaseNumber", "Depression_18", "Chronic_pain", "ABIS",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "CORT/ACTH", "CORT/IL6", "CORT/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]

OUTCOME = "Anxiety_14"
RANDOM_STATE = 284

# =============================================================================
# 高斯噪声增强函数
# =============================================================================

def gaussian_noise_augmentation(
    X, 
    y, 
    noise_factor=0.1, 
    n_augment=2, 
    feature_noise=None, 
    minority_only=False,
    random_state=284
):
    """
    高斯噪声数据增强
    
    参数:
    - X: 特征矩阵 (DataFrame)
    - y: 标签 (Series)
    - noise_factor: 噪声强度系数 (默认 0.1)
    - n_augment: 增强倍数，不含原始样本 (默认 2，即总共 3 倍样本)
    - feature_noise: 各特征的自定义噪声强度字典 {特征名: 强度}
    - minority_only: 是否仅增强少数类 (默认 False)
    - random_state: 随机种子
    
    返回:
    - X_augmented: 增强后的特征矩阵
    - y_augmented: 增强后的标签
    - augmentation_info: 增强信息字典
    """
    np.random.seed(random_state)
    
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    
    # 计算噪声强度统计
    noise_stats = {}
    
    for aug_idx in range(n_augment):
        X_noisy = X.copy()
        
        for col in X.columns:
            # 确定噪声强度
            if feature_noise and col in feature_noise:
                scale = feature_noise[col] * X[col].std()
            else:
                scale = noise_factor * X[col].std()
            
            # 记录噪声统计
            if col not in noise_stats:
                noise_stats[col] = {"scale": scale, "std_ratio": noise_factor if feature_noise is None else feature_noise.get(col, noise_factor)}
            
            # 仅增强少数类或全部样本
            if minority_only:
                minority_mask = (y == 1).values
                noise = np.random.normal(0, scale, size=len(X))
                noise[~minority_mask] = 0  # 多数类不添加噪声
            else:
                noise = np.random.normal(0, scale, size=len(X))
            
            X_noisy[col] = X[col].values + noise
        
        X_augmented.append(X_noisy)
        y_augmented.append(y.copy())
    
    augmentation_info = {
        "n_original": len(X),
        "n_augmented": len(X) * (n_augment + 1),
        "augment_ratio": n_augment + 1,
        "noise_factor": noise_factor,
        "feature_noise": feature_noise,
        "minority_only": minority_only,
        "noise_stats": noise_stats,
    }
    
    return pd.concat(X_augmented, ignore_index=True), \
           pd.concat(y_augmented, ignore_index=True), \
           augmentation_info


# =============================================================================
# 辅助函数
# =============================================================================

def compute_youden_threshold(y_true, y_prob):
    """计算 Youden 指数最优阈值"""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values

    idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[idx]
    y_prob_sorted = y_prob[idx]

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.arange(1, n + 1) - tp_cum

    sensitivity = tp_cum / n_pos
    specificity = (n_neg - fp_cum) / n_neg
    youden = sensitivity + specificity - 1

    best_idx = np.argmax(youden)
    best_thresh = y_prob_sorted[best_idx]
    return best_thresh, youden[best_idx]


def oof_threshold_selection_any(model_func, X, y, n_splits=5):
    """通用 OOF 阈值选择"""
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
    """评估模型性能"""
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


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name, model_func, features=None):
    """
    训练并评估单个模型
    
    参数:
    - X_train, y_train: 训练数据
    - X_test, y_test: 测试数据
    - model_name: 模型名称
    - model_func: 创建模型的函数
    - features: 使用的特征列表 (None 表示使用全部)
    
    返回:
    - metrics: 性能指标字典
    - threshold: 阈值
    - model: 训练好的模型
    """
    if features is not None:
        X_tr = X_train[features] if isinstance(X_train, pd.DataFrame) else X_train[:, features]
        X_te = X_test[features] if isinstance(X_test, pd.DataFrame) else X_test[:, features]
    else:
        X_tr = X_train
        X_te = X_test
    
    # OOF 阈值选择
    _, threshold = oof_threshold_selection_any(model_func, pd.DataFrame(X_tr) if not isinstance(X_tr, pd.DataFrame) else X_tr, y_train)
    
    # 训练最终模型
    model = model_func()
    model.fit(X_tr, y_train)
    
    # 测试集预测
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["Model"] = model_name
    metrics["Threshold"] = round(threshold, 4)
    
    return metrics, threshold, model


# =============================================================================
# 主程序
# =============================================================================

print("=" * 70)
print("Step 3 + 高斯噪声增强：性能对比实验")
print("=" * 70)

# 读取数据
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"训练文件未找到: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"测试文件未找到: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 验证列
for col in [OUTCOME] + BIOMARKERS:
    if col not in train_df.columns:
        raise ValueError(f"训练数据缺少列: {col}")
    if col not in test_df.columns:
        raise ValueError(f"测试数据缺少列: {col}")

# 特征矩阵和标签
X_train = train_df[BIOMARKERS].copy()
y_train = train_df[OUTCOME].copy()
X_test = test_df[BIOMARKERS].copy()
y_test = test_df[OUTCOME].copy()

print(f"\n原始训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
print(f"训练集焦虑患病率: {y_train.mean():.4f}")
print(f"测试集焦虑患病率: {y_test.mean():.4f}")
print(f"生物标志物: {BIOMARKERS}")
print(f"XGBoost 可用: {HAS_XGBOOST}")

# 创建输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 高斯噪声增强配置
# =============================================================================

# 特征自适应噪声强度（基于生物标志物特性）
FEATURE_NOISE_SCALES = {
    "IL6": 0.10,       # 细胞因子，个体变异大
    "IL10": 0.10,      # 细胞因子
    "TNFalpha": 0.10,  # 细胞因子
    "CRP": 0.15,       # 急性期蛋白，日变异大
    "ACTH": 0.08,      # 激素
    "CORT": 0.08,      # 激素
}

# 实验配置
EXPERIMENTS = [
    {
        "name": "原始数据（无增强）",
        "augment": False,
        "noise_factor": None,
        "feature_noise": None,
        "minority_only": False,
    },
    {
        "name": "基础高斯噪声 (α=0.05)",
        "augment": True,
        "noise_factor": 0.05,
        "feature_noise": None,
        "minority_only": False,
    },
    {
        "name": "基础高斯噪声 (α=0.10)",
        "augment": True,
        "noise_factor": 0.10,
        "feature_noise": None,
        "minority_only": False,
    },
    {
        "name": "特征自适应噪声",
        "augment": True,
        "noise_factor": None,
        "feature_noise": FEATURE_NOISE_SCALES,
        "minority_only": False,
    },
    {
        "name": "少数类增强（特征自适应）",
        "augment": True,
        "noise_factor": None,
        "feature_noise": FEATURE_NOISE_SCALES,
        "minority_only": True,
    },
]

# =============================================================================
# 运行实验
# =============================================================================

all_results = []
all_augmentation_info = []

print("\n" + "=" * 70)
print("开始实验...")
print("=" * 70)

for exp_idx, exp in enumerate(EXPERIMENTS):
    print(f"\n--- 实验 {exp_idx + 1}/{len(EXPERIMENTS)}: {exp['name']} ---")
    
    # 数据准备
    if exp["augment"]:
        X_train_aug, y_train_aug, aug_info = gaussian_noise_augmentation(
            X_train, y_train,
            noise_factor=exp["noise_factor"] if exp["noise_factor"] else 0.1,
            n_augment=2,  # 增强 2 倍，总共 3 倍样本
            feature_noise=exp["feature_noise"],
            minority_only=exp["minority_only"],
            random_state=RANDOM_STATE,
        )
        aug_info["experiment_name"] = exp["name"]
        all_augmentation_info.append(aug_info)
        print(f"  增强后训练样本: {len(X_train_aug)} (原始: {len(X_train)})")
    else:
        X_train_aug = X_train.copy()
        y_train_aug = y_train.copy()
        aug_info = {"experiment_name": exp["name"], "n_original": len(X_train), "n_augmented": len(X_train)}
        all_augmentation_info.append(aug_info)
        print(f"  使用原始训练样本: {len(X_train_aug)}")
    
    # 存储该实验的所有模型结果
    exp_results = []
    
    # -------------------------------------------------------------------------
    # 单一生物标志物模型（仅最佳：CRP）
    # -------------------------------------------------------------------------
    print(f"  训练单一指标 LR (CRP)...")
    
    def make_lr_crp():
        return LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )
    
    crp_metrics, crp_thresh, crp_model = train_and_evaluate_model(
        X_train_aug, y_train_aug, X_test, y_test,
        f"CRP_LR",
        make_lr_crp,
        features=["CRP"],
    )
    crp_metrics["Experiment"] = exp["name"]
    crp_metrics["Model_Type"] = "Single"
    exp_results.append(crp_metrics)
    print(f"    ROC_AUC: {crp_metrics['ROC_AUC']:.4f}, PR_AUC: {crp_metrics['PR_AUC']:.4f}")
    
    # -------------------------------------------------------------------------
    # 六指标 LASSO
    # -------------------------------------------------------------------------
    print(f"  训练六指标 LASSO...")
    
    # GridSearchCV 找最佳 C
    lasso_base = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lasso_cv = GridSearchCV(lasso_base, param_grid, cv=skf, scoring="roc_auc", refit=True)
    lasso_cv.fit(X_train_aug, y_train_aug)
    best_C = lasso_cv.best_params_["C"]
    
    def make_lasso():
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=best_C,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )
    
    lasso_metrics, lasso_thresh, lasso_model = train_and_evaluate_model(
        X_train_aug, y_train_aug, X_test, y_test,
        "Six_LASSO",
        make_lasso,
    )
    lasso_metrics["Experiment"] = exp["name"]
    lasso_metrics["Model_Type"] = "Six"
    lasso_metrics["Best_C"] = best_C
    exp_results.append(lasso_metrics)
    print(f"    Best C: {best_C}, ROC_AUC: {lasso_metrics['ROC_AUC']:.4f}")
    
    # -------------------------------------------------------------------------
    # 六指标 Random Forest
    # -------------------------------------------------------------------------
    print(f"  训练六指标 Random Forest...")
    
    def make_rf():
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    
    rf_metrics, rf_thresh, rf_model = train_and_evaluate_model(
        X_train_aug, y_train_aug, X_test, y_test,
        "Six_RF",
        make_rf,
    )
    rf_metrics["Experiment"] = exp["name"]
    rf_metrics["Model_Type"] = "Six"
    exp_results.append(rf_metrics)
    print(f"    ROC_AUC: {rf_metrics['ROC_AUC']:.4f}, PR_AUC: {rf_metrics['PR_AUC']:.4f}")
    
    # -------------------------------------------------------------------------
    # 六指标 XGBoost
    # -------------------------------------------------------------------------
    if HAS_XGBOOST:
        print(f"  训练六指标 XGBoost...")
        
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
        
        xgb_metrics, xgb_thresh, xgb_model = train_and_evaluate_model(
            X_train_aug, y_train_aug, X_test, y_test,
            "Six_XGBoost",
            make_xgb,
        )
        xgb_metrics["Experiment"] = exp["name"]
        xgb_metrics["Model_Type"] = "Six"
        exp_results.append(xgb_metrics)
        print(f"    ROC_AUC: {xgb_metrics['ROC_AUC']:.4f}, PR_AUC: {xgb_metrics['PR_AUC']:.4f}")
    
    all_results.extend(exp_results)

# =============================================================================
# 结果汇总
# =============================================================================

results_df = pd.DataFrame(all_results)

# 重新排列列
col_order = [
    "Experiment", "Model", "Model_Type", "ROC_AUC", "PR_AUC", 
    "Accuracy", "Sensitivity", "Specificity", "Precision", "F1",
    "Threshold", "Best_C", "TN", "FP", "FN", "TP"
]
col_order = [c for c in col_order if c in results_df.columns]
results_df = results_df[col_order]

# 保存结果
results_df.to_csv(OUT_DIR / "table_augmentation_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n保存结果: {OUT_DIR / 'table_augmentation_comparison.csv'}")

# 保存增强信息
aug_info_df = pd.DataFrame(all_augmentation_info)
aug_info_df.to_csv(OUT_DIR / "augmentation_info.csv", index=False, encoding="utf-8-sig")
print(f"保存增强信息: {OUT_DIR / 'augmentation_info.csv'}")

# =============================================================================
# 性能对比表格（按实验和模型类型）
# =============================================================================

print("\n" + "=" * 70)
print("性能对比汇总")
print("=" * 70)

# 创建对比表格
comparison_table = results_df.pivot_table(
    index="Experiment",
    columns="Model",
    values="ROC_AUC",
    aggfunc="first"
).round(4)

print("\nROC-AUC 对比:")
print(comparison_table.to_string())

# 保存对比表格
comparison_table.to_csv(OUT_DIR / "table_roc_auc_comparison.csv", encoding="utf-8-sig")

# =============================================================================
# 可视化
# =============================================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

# --- 图 1: ROC-AUC 柱状图对比 ---
fig, ax = plt.subplots(figsize=(12, 6))

models = ["CRP_LR", "Six_LASSO", "Six_RF", "Six_XGBoost"]
experiments = [exp["name"] for exp in EXPERIMENTS]

x = np.arange(len(models))
width = 0.15
colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

for i, exp_name in enumerate(experiments):
    exp_data = results_df[results_df["Experiment"] == exp_name]
    aucs = [exp_data[exp_data["Model"] == m]["ROC_AUC"].values[0] 
            if len(exp_data[exp_data["Model"] == m]) > 0 else 0 
            for m in models]
    bars = ax.bar(x + i * width, aucs, width, label=exp_name, color=colors[i])
    # 添加数值标签
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xlabel("模型", fontsize=12)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("高斯噪声增强性能对比 — ROC-AUC", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * (len(experiments) - 1) / 2)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim([0, 1.0])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure_augmentation_roc_auc_comparison.png", dpi=300)
plt.close(fig)
print(f"\n保存图表: {OUT_DIR / 'figure_augmentation_roc_auc_comparison.png'}")

# --- 图 2: 原始 vs 最佳增强对比 ---
fig, ax = plt.subplots(figsize=(8, 5))

# 找出最佳增强方法
baseline_exp = "原始数据（无增强）"
baseline_data = results_df[results_df["Experiment"] == baseline_exp]

# 计算各增强方法相对基线的提升
improvements = []
for exp_name in experiments[1:]:  # 跳过原始数据
    exp_data = results_df[results_df["Experiment"] == exp_name]
    for model in models:
        base_auc = baseline_data[baseline_data["Model"] == model]["ROC_AUC"].values[0] \
            if len(baseline_data[baseline_data["Model"] == model]) > 0 else 0
        exp_auc = exp_data[exp_data["Model"] == model]["ROC_AUC"].values[0] \
            if len(exp_data[exp_data["Model"] == model]) > 0 else 0
        improvement = exp_auc - base_auc
        improvements.append({
            "Experiment": exp_name,
            "Model": model,
            "Baseline_AUC": base_auc,
            "Augmented_AUC": exp_auc,
            "Improvement": improvement,
        })

improve_df = pd.DataFrame(improvements)

# 找出最佳增强方法
best_aug_exp = improve_df.groupby("Experiment")["Improvement"].mean().idxmax()
print(f"\n最佳增强方法（平均 ROC-AUC 提升）: {best_aug_exp}")

# 绘制原始 vs 最佳增强对比
best_data = results_df[results_df["Experiment"] == best_aug_exp]

x = np.arange(len(models))
width = 0.35

baseline_aucs = [baseline_data[baseline_data["Model"] == m]["ROC_AUC"].values[0] 
                 if len(baseline_data[baseline_data["Model"] == m]) > 0 else 0 
                 for m in models]
best_aucs = [best_data[best_data["Model"] == m]["ROC_AUC"].values[0] 
             if len(best_data[best_data["Model"] == m]) > 0 else 0 
             for m in models]

bars1 = ax.bar(x - width/2, baseline_aucs, width, label=baseline_exp, color="#1f77b4")
bars2 = ax.bar(x + width/2, best_aucs, width, label=best_aug_exp, color="#ff7f0e")

# 添加数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel("模型", fontsize=12)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title(f"原始 vs 最佳增强方法对比\n({best_aug_exp})", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim([0, 1.0])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "figure_baseline_vs_best_augmentation.png", dpi=300)
plt.close(fig)
print(f"保存图表: {OUT_DIR / 'figure_baseline_vs_best_augmentation.png'}")

# =============================================================================
# 保存增强后的训练数据（最佳方法）
# =============================================================================

# 使用最佳增强方法重新生成增强数据
best_exp_config = next(exp for exp in EXPERIMENTS if exp["name"] == best_aug_exp)
X_train_best, y_train_best, _ = gaussian_noise_augmentation(
    X_train, y_train,
    noise_factor=best_exp_config["noise_factor"] if best_exp_config["noise_factor"] else 0.1,
    n_augment=2,
    feature_noise=best_exp_config["feature_noise"],
    minority_only=best_exp_config["minority_only"],
    random_state=RANDOM_STATE,
)

# 保存增强后的训练数据
augmented_train_df = pd.concat([
    X_train_best.reset_index(drop=True),
    y_train_best.reset_index(drop=True)
], axis=1)
augmented_train_df.to_csv(OUT_DIR / "train_augmented_best.csv", index=False)
print(f"\n保存最佳增强训练数据: {OUT_DIR / 'train_augmented_best.csv'}")
print(f"  样本数: {len(augmented_train_df)} (原始: {len(X_train)})")

# =============================================================================
# 日志
# =============================================================================

log_lines = []
log_lines.append("=" * 70)
log_lines.append("Step 3 + 高斯噪声增强 日志")
log_lines.append("=" * 70)
log_lines.append(f"输入训练集: {TRAIN_PATH}")
log_lines.append(f"输入测试集: {TEST_PATH}")
log_lines.append(f"原始训练样本: {len(X_train)}")
log_lines.append(f"测试样本: {len(X_test)}")
log_lines.append(f"生物标志物: {BIOMARKERS}")
log_lines.append(f"XGBoost 可用: {HAS_XGBOOST}")
log_lines.append("")
log_lines.append("=" * 70)
log_lines.append("实验配置:")
log_lines.append("=" * 70)
for exp in EXPERIMENTS:
    log_lines.append(f"  - {exp['name']}")
    if exp['augment']:
        log_lines.append(f"    噪声强度: {exp['noise_factor']}")
        log_lines.append(f"    特征自适应: {exp['feature_noise'] is not None}")
        log_lines.append(f"    仅少数类: {exp['minority_only']}")
log_lines.append("")
log_lines.append("=" * 70)
log_lines.append("性能汇总 (ROC-AUC):")
log_lines.append("=" * 70)
log_lines.append(comparison_table.to_string())
log_lines.append("")
log_lines.append(f"最佳增强方法: {best_aug_exp}")
log_lines.append(f"增强后训练样本: {len(X_train_best)}")
log_lines.append("")
log_lines.append("=" * 70)
log_lines.append("各模型性能详情:")
log_lines.append("=" * 70)
for exp_name in experiments:
    log_lines.append(f"\n--- {exp_name} ---")
    exp_data = results_df[results_df["Experiment"] == exp_name]
    for _, row in exp_data.iterrows():
        log_lines.append(f"  {row['Model']}:")
        log_lines.append(f"    ROC_AUC: {row['ROC_AUC']:.4f}")
        log_lines.append(f"    PR_AUC:  {row['PR_AUC']:.4f}")
        log_lines.append(f"    Sens: {row['Sensitivity']:.4f}, Spec: {row['Specificity']:.4f}")
        log_lines.append(f"    F1: {row['F1']:.4f}, Acc: {row['Accuracy']:.4f}")
log_lines.append("")
log_lines.append("=" * 70)
log_lines.append("输出文件:")
log_lines.append("=" * 70)
log_lines.append(f"  - table_augmentation_comparison.csv")
log_lines.append(f"  - table_roc_auc_comparison.csv")
log_lines.append(f"  - augmentation_info.csv")
log_lines.append(f"  - figure_augmentation_roc_auc_comparison.png")
log_lines.append(f"  - figure_baseline_vs_best_augmentation.png")
log_lines.append(f"  - train_augmented_best.csv")
log_lines.append("")

with open(OUT_DIR / "step3_gaussian_augmentation_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n保存日志: {OUT_DIR / 'step3_gaussian_augmentation_log.txt'}")

print("\n" + "=" * 70)
print("Step 3 + 高斯噪声增强 完成!")
print("=" * 70)
print(f"最佳增强方法: {best_aug_exp}")
print(f"所有结果保存在: {OUT_DIR}")