"""比率计算模块。

提供血液学比率的重新计算和验证功能，
确保比率值的准确性和一致性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def recompute_ratios(
    frame: pd.DataFrame, formulas: dict[str, dict[str, str]], *, denominator_epsilon: float = 1e-12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """重新计算并验证比率特征。

    根据公式定义重新计算比率值，并与原始存储值进行比对验证。
    对于分母接近零的情况进行安全处理。

    Args:
        frame: 包含原始特征和比率特征的数据框。
        formulas: 比率公式字典，键为比率名称，值为包含numerator和denominator的字典。
        denominator_epsilon: 分母安全阈值，小于此值视为接近零。

    Returns:
        元组，包含清洗后的数据框（使用重新计算的比率值）和验证结果数据框。
        验证结果包含每行的存储值、重算值、误差等信息。
    """
    cleaned = frame.copy()
    rows: list[dict] = []
    for ratio, formula in formulas.items():
        numerator, denominator = formula["numerator"], formula["denominator"]
        safe_denominator = cleaned[denominator].where(cleaned[denominator].abs() > denominator_epsilon)
        calculated = cleaned[numerator] / safe_denominator
        stored = cleaned[ratio] if ratio in cleaned else pd.Series(np.nan, index=cleaned.index)
        absolute_error = (stored - calculated).abs()
        relative_error = absolute_error / calculated.abs().clip(lower=denominator_epsilon)
        rows.extend(
            {
                "row_index": int(index),
                "ratio": ratio,
                "stored": stored.loc[index],
                "recalculated": calculated.loc[index],
                "absolute_error": absolute_error.loc[index],
                "relative_error": relative_error.loc[index],
                "denominator_near_zero": bool(pd.isna(safe_denominator.loc[index])),
            }
            for index in cleaned.index
        )
        cleaned[ratio] = calculated
    return cleaned, pd.DataFrame(rows)
