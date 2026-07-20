"""数据桥接模块。

提供数据清洗和多视图构建的统一入口，
协调比率重算和视图构建操作。
"""

from __future__ import annotations

import pandas as pd

from .ratios import recompute_ratios
from .views import build_views


def hematological_bridge(frame: pd.DataFrame, dataset_config: dict, formulas: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """血液学数据桥接函数。

    执行数据清洗、比率重算和多视图构建的完整流程。

    Args:
        frame: 原始数据框。
        dataset_config: 数据集配置字典，包含raw_features和ratio_features。
        formulas: 比率公式字典，定义各比率的计算方式。

    Returns:
        元组，包含清洗后的数据框、多视图字典和验证结果数据框。
    """
    cleaned, validation = recompute_ratios(frame, formulas)
    views = build_views(cleaned, dataset_config["raw_features"], dataset_config["ratio_features"])
    return cleaned, views, validation
