"""PySR符号回归学生模型模块。

提供基于PySR的符号回归功能，
用于从模型预测中学习可解释的数学表达式。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logit


@dataclass(frozen=True)
class PySRCapability:
    """PySR能力检查结果数据类。

    Attributes:
        available: PySR是否可用。
        reason: 不可用原因或可用说明。
    """

    available: bool
    reason: str


def configure_julia() -> str | None:
    """配置Julia运行时环境。

    查找并配置PySR所需的Julia执行路径，
    优先使用用户本地安装的Julia。

    Returns:
        Julia可执行文件路径，未找到时返回None。
    """
    configured = os.getenv("PYTHON_JULIAPKG_EXE")
    if configured:
        return configured
    local_julia = Path.home() / ".local" / "bin" / "julia"
    if local_julia.is_file():
        os.environ["PYTHON_JULIAPKG_EXE"] = str(local_julia)
        return str(local_julia)
    return None


def capability() -> PySRCapability:
    """检查PySR后端能力。

    验证pysr包、juliacall后端和Julia运行时的可用性。

    Returns:
        PySR能力检查结果。
    """
    if importlib.util.find_spec("pysr") is None:
        return PySRCapability(False, "pysr is not installed; use uv sync --extra symbolic")
    if importlib.util.find_spec("juliacall") is None:
        return PySRCapability(False, "juliacall backend is missing")
    executable = configure_julia()
    if executable is None:
        return PySRCapability(False, "Julia executable is not configured")
    try:
        version = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return PySRCapability(False, f"Julia runtime check failed: {exc}")
    return PySRCapability(True, f"PySR frontend and {version} detected")


def train_pysr(
    features,
    target_probability,
    *,
    niterations: int = 100,
    output_directory: str | None = None,
    run_id: str | None = None,
    seed: int = 284,
    maxsize: int = 20,
    maxdepth: int = 5,
):
    """训练PySR符号回归模型。

    使用符号回归从特征和目标概率中学习数学表达式，
    生成可解释的预测公式。

    Args:
        features: 训练特征数据框。
        target_probability: 目标概率数组，通常来自教师模型。
        niterations: 符号回归迭代次数。
        output_directory: 输出目录路径，用于保存发现的方程。
        run_id: 运行标识符，用于区分不同运行。
        seed: 随机种子。
        maxsize: 表达式最大尺寸。
        maxdepth: 表达式最大深度。

    Returns:
        训练好的PySRRegressor模型实例。

    Raises:
        RuntimeError: PySR或Julia环境不可用时抛出。
    """
    status = capability()
    if not status.available:
        raise RuntimeError(status.reason)
    configure_julia()
    from pysr import PySRRegressor

    target = logit(np.clip(np.asarray(target_probability, dtype=float), 1e-6, 1 - 1e-6))
    model = PySRRegressor(
        niterations=niterations, binary_operators=["+", "-", "*", "/"], unary_operators=["log", "sqrt"],
        maxsize=maxsize, maxdepth=maxdepth, model_selection="best", output_directory=output_directory,
        run_id=run_id, parallelism="serial", populations=4, population_size=24,
        progress=False, verbosity=0, random_state=seed, deterministic=True,
    )
    return model.fit(features, target)
