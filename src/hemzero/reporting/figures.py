"""图表生成模块。

提供实验结果可视化功能，
生成符合出版要求的图表文件。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_auroc(metrics: pd.DataFrame, output: Path) -> None:
    """绘制AUROC对比条形图。

    生成模型AUROC指标的水平条形图，
    包含随机基线参考线。

    Args:
        metrics: 模型指标数据框，需包含model和AUROC列。
        output: 输出图片文件路径。
    """
    ordered = metrics.sort_values("AUROC")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(ordered["model"], ordered["AUROC"], color="#3572A5")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlabel="AUROC", xlim=(0, 1))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
