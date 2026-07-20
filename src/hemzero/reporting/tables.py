"""表格生成模块。

提供实验结果表格的格式化和导出功能，
生成符合出版要求的汇总表格。
"""

from __future__ import annotations

import pandas as pd


def model_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """生成模型对比表格。

    从原始指标数据框中提取关键指标并按AUROC排序，
    生成便于对比的格式化表格。

    Args:
        metrics: 原始模型指标数据框。

    Returns:
        格式化的模型对比表格，按AUROC降序排列。
    """
    columns = [column for column in ["model", "AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity", "F1", "MCC"] if column in metrics]
    return metrics[columns].sort_values("AUROC", ascending=False).reset_index(drop=True)
