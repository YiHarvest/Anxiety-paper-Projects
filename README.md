# 焦虑血液生物标志物二分类预测

基于外周血生物标志物的焦虑二分类预测实验。  
结局变量：`Anxiety_14` | 随机种子：`random_state=42` | 数据集：481 例（训练 336，测试 145）

## 项目结构

```
D:\sleep\AnxietyProjects\
├── dataset\dataset\              # 原始数据
│   ├── anxiety_bio15_train.csv
│   └── anxiety_bio15_test.csv
├── pyproject.toml                # 依赖 + 阿里云镜像
├── step2_preprocess_abis.py      # Step 2：预处理 + ABIS 计算
├── step3_single_six_models.py    # Step 3：单一指标 + 六指标模型
├── step4_ratio_integrated_models.py  # Step 4：比值 + 整合模型
├── step5_abis_bootstrap_compare.py   # Step 5：ABIS + Bootstrap + 核心对照
├── step6_model_interpretation.py     # Step 6：模型解释（SHAP）
├── step7_final_summary.py            # Step 7：最终汇总
├── step8_calibration_dca_sensitivity.py  # Step 8：校准 + DCA + 稳健性
├── README.md
└── output\                        # 全部输出
    ├── step2_preprocess_abis/
    ├── step3_single_six_models/
    ├── step4_ratio_integrated_models/
    ├── step5_abis_bootstrap_compare/
    ├── step6_model_interpretation/
    ├── step7_final_summary/
    └── step8_calibration_dca_sensitivity/
```

## 快速开始（3 步）

```powershell
# ① 安装 uv（如果没有）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# ② 进入项目 + 一键安装依赖（阿里云国内镜像，下载飞快）
cd D:\sleep\AnxietyProjects
uv sync

# ③ 一键运行全部（Step 2 ~ Step 8，约 20-30 分钟）
uv run python step2_preprocess_abis.py; uv run python step3_single_six_models.py; uv run python step4_ratio_integrated_models.py; uv run python step5_abis_bootstrap_compare.py; uv run python step6_model_interpretation.py; uv run python step7_final_summary.py; uv run python step8_calibration_dca_sensitivity.py
```

```powershell
cd D:\sleep\AnxietyProjects

uv run python step2_preprocess_abis.py

uv run python step3_single_six_models.py

uv run python step4_ratio_integrated_models.py

uv run python step5_abis_bootstrap_compare.py

uv run python step6_model_interpretation.py

uv run python step7_final_summary.py

uv run python step8_calibration_dca_sensitivity.py
```

## 各步骤说明

| Step | 脚本 | 内容 | 耗时 |
|------|------|------|------|
| 2 | `step2_preprocess_abis.py` | 预处理：中位数填补→log1p→winsorization→z-score→ABIS 计算 | < 30s |
| 3 | `step3_single_six_models.py` | 6 个单一指标 LR + Six-LASSO/Six-RF/Six-XGBoost 模型 | < 30s |
| 4 | `step4_ratio_integrated_models.py` | 9 项比值模型 + 15 项整合模型 (LASSO/RF/XGBoost) | < 1min |
| 5 | `step5_abis_bootstrap_compare.py` | ABIS LR + Bootstrap 1000 CI + 核心模型对照表 | ~3min |
| 6 | `step6_model_interpretation.py` | Built-in + Permutation + SHAP 特征重要性 | ~5min |
| 7 | `step7_final_summary.py` | 最终汇总表、核心图表、论文 Results/Discussion 文本 | < 30s |
| 8 | `step8_calibration_dca_sensitivity.py` | 校准曲线、DCA、1000 次重复随机划分稳健性 | ~15-30min |

## 依赖说明

所有依赖在 `pyproject.toml` 中统一管理，`uv sync` 一键安装：

| 包 | 版本 | 用途 |
|----|------|------|
| pandas | ≥2.0 | 数据处理 |
| numpy | ≥1.24 | 数值计算 |
| scipy | ≥1.10 | 统计检验 |
| scikit-learn | ≥1.3 | 机器学习模型 |
| matplotlib | ≥3.7 | 可视化 |
| xgboost | ≥2.0 | 梯度提升模型 |
| shap | ≥0.42 | 模型解释 |

## 配置

- **结局变量**：`Anxiety_14`（二分类，1=焦虑阳性）
- **随机种子**：`random_state=42`
- **PyPI 镜像**：阿里云 `mirrors.aliyun.com/pypi/simple/`
- **六项原始血液指标**：IL6、IL10、TNFalpha、CRP、ACTH、CORT
- **九项比值指标**：IL6/IL10、TNFalpha/IL10、CRP/IL10、CORT/ACTH、CORT/IL6、CORT/CRP、IL6/TNFalpha、CRP/IL6、ACTH/IL6
- **排他特征**：CaseNumber、Depression_18、Chronic_pain（不纳入模型）
- 所有脚本必须从项目根目录 `D:\sleep\AnxietyProjects` 运行


