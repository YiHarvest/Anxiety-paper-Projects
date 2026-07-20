"""实验跟踪模块。

提供MLflow集成的实验跟踪功能，
支持可选的实验运行记录和参数日志。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def tracked_run(*, run_id: str, tracking_uri: Path, parameters: dict):
    """创建跟踪的实验运行上下文。

    使用MLflow记录实验运行，包括参数和指标。
    当MLflow未安装时静默跳过跟踪。

    Args:
        run_id: 运行标识符。
        tracking_uri: MLflow跟踪数据库路径。
        parameters: 要记录的参数字典。

    Yields:
        MLflow活动运行对象，或MLflow未安装时为None。
    """
    try:
        import mlflow
    except ImportError:
        yield None
        return
    tracking_uri = tracking_uri.resolve()
    tracking_uri.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")
    mlflow.set_experiment("HemoZero")
    with mlflow.start_run(run_name=run_id) as active:
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        yield active
