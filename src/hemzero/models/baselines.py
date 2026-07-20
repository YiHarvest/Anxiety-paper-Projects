"""基线模型模块。

提供传统机器学习基线模型的工厂函数，
包括逻辑回归、弹性网络、SVM和随机森林。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


def baseline_factories(seed: int = 284) -> dict[str, callable]:
    """创建基线模型工厂字典。

    返回包含多种基线分类器的工厂函数字典，
    每个工厂函数返回一个配置好的模型实例。

    Args:
        seed: 随机种子，用于确保可重复性。

    Returns:
        模型工厂字典，包含：
        - 'logistic': 逻辑回归模型
        - 'elastic_net': 弹性网络逻辑回归模型
        - 'svm': 支持向量机模型（带概率校准）
        - 'random_forest': 随机森林模型
    """

    def linear(**kwargs):
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
             ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed, **kwargs))]
        )

    return {
        "logistic": lambda: linear(),
        "elastic_net": lambda: linear(solver="saga", l1_ratio=0.5, C=0.5),
        "svm": lambda: Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
             ("model", CalibratedClassifierCV(
                 SVC(class_weight="balanced", random_state=seed), cv=3, ensemble=False
             ))]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
    }
