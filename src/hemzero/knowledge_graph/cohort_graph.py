"""Fold-local, association-only cohort graph calculations."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

from .lineage_sync import canonical_pair, feature_entity_id


def _partial_correlation(frame: pd.DataFrame, epsilon: float) -> np.ndarray:
    covariance = np.cov(frame.to_numpy(float), rowvar=False)
    precision = np.linalg.pinv(covariance + epsilon * np.eye(covariance.shape[0]))
    diagonal = np.sqrt(np.maximum(np.diag(precision), epsilon))
    return -precision / np.outer(diagonal, diagonal)


def build_fold_associations(
    frame: pd.DataFrame,
    *,
    run_id: str,
    repeat: int,
    fold: int,
    spearman_min_absolute: float,
    partial_min_absolute: float,
    graphical_lasso_alpha: float,
    epsilon: float = 1e-8,
) -> list[dict]:
    """Calculate association edges from one outer-training fold only."""

    clean = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(frame.median())
    scaled = (clean - clean.mean()) / clean.std(ddof=0).replace(0, 1)
    spearman = clean.corr(method="spearman").to_numpy(float)
    partial = _partial_correlation(scaled, epsilon)
    graphical = GraphicalLasso(alpha=graphical_lasso_alpha, max_iter=500).fit(scaled.to_numpy(float))
    diagonal = np.sqrt(np.maximum(np.diag(graphical.precision_), epsilon))
    glasso_partial = -graphical.precision_ / np.outer(diagonal, diagonal)
    rows: list[dict] = []
    names = list(clean.columns)
    for left_index, right_index in combinations(range(len(names)), 2):
        feature_a, feature_b = names[left_index], names[right_index]
        pair_id = canonical_pair(feature_a, feature_b)
        values = (
            ("spearman", "CORRELATED_WITH", spearman[left_index, right_index], spearman_min_absolute),
            ("partial_correlation", "CONDITIONALLY_ASSOCIATED_WITH", partial[left_index, right_index], partial_min_absolute),
            ("graphical_lasso", "CONDITIONALLY_ASSOCIATED_WITH", glasso_partial[left_index, right_index], partial_min_absolute),
        )
        for method, relation, value, threshold in values:
            if np.isfinite(value) and abs(float(value)) >= threshold:
                rows.append({
                    "association_id": f"{run_id}__r{repeat}__f{fold}__{pair_id}__{method}",
                    "run_id": run_id, "repeat": repeat, "fold": fold, "pair_id": pair_id,
                    "feature_a": feature_a, "feature_b": feature_b,
                    "feature_a_id": feature_entity_id(feature_a), "feature_b_id": feature_entity_id(feature_b),
                    "relationship_type": relation, "method": method, "value": float(value),
                    "absolute_value": abs(float(value)), "source_partition": "outer_training_fold",
                    "uses_target": False,
                })
    return rows
