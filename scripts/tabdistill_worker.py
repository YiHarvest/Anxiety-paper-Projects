#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--target-column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-names", required=True)
    parser.add_argument("--sample-budget", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--index-type", default="fbii")
    parser.add_argument("--index-types")
    parser.add_argument("--seed", type=int, default=284)
    arguments = parser.parse_args()
    feature_names = json.loads(arguments.feature_names)
    if arguments.training_data is not None:
        # Import TabPFN before adding TabDistill's top-level `src` package to sys.path.
        from tabpfn import TabPFNClassifier

        frame = pd.read_csv(arguments.training_data)
        development, explanation = train_test_split(
            frame, test_size=0.25, stratify=frame[arguments.target_column], random_state=arguments.seed
        )
        teacher = TabPFNClassifier(
            device="cpu", n_estimators=1, auto_scale_n_estimators=False,
            fit_mode="low_memory", memory_saving_mode=True, keep_cache_on_device=False,
            n_preprocessing_jobs=1, show_progress_bar=False, random_state=arguments.seed,
        )
        teacher.fit(development[feature_names], development[arguments.target_column])
        features = explanation[feature_names].to_numpy(float)
    else:
        if arguments.teacher_model is None or arguments.features is None:
            raise ValueError("Provide either --training-data or both --teacher-model and --features")
        teacher = joblib.load(arguments.teacher_model)
        features = pd.read_csv(arguments.features)[feature_names].to_numpy(float)
    sys.path.insert(0, str(arguments.repository))
    from src.spectralexplain.explainer import Explainer

    baseline = np.nanmean(features, axis=0)
    index_types = json.loads(arguments.index_types) if arguments.index_types else [arguments.index_type]
    aggregate: dict[tuple[str, tuple[str, str]], list[float]] = defaultdict(list)
    for point in features[: arguments.max_samples]:
        def value_function(mask):
            mask = np.asarray(mask, dtype=bool)
            if mask.ndim == 1:
                mask = mask[None, :]
            samples = pd.DataFrame(np.where(mask, point, baseline), columns=feature_names)
            probability = np.clip(teacher.predict_proba(samples)[:, 1], 1e-10, 1 - 1e-10)
            return np.log(probability / (1 - probability))

        explainer = Explainer(value_function=value_function, features=range(len(feature_names)),
                             sample_budget=arguments.sample_budget, max_order=2)
        for index_type in index_types:
            result = explainer.interactions(index_type)
            for indices, score in result.interactions.items():
                if len(indices) == 2:
                    pair = tuple(feature_names[index] for index in indices)
                    aggregate[(index_type, pair)].append(float(score))
    candidates = [
        {"index_type": index_type, "features": list(pair),
         "frequency": len(scores) / min(arguments.max_samples, len(features)),
         "score": float(np.mean(scores)), "direction_consistency": float(abs(np.mean(np.sign(scores))))}
        for (index_type, pair), scores in aggregate.items()
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({
        "index_types": index_types, "candidates": candidates,
        "teacher_backend": "TabPFN-3", "fold_local": arguments.training_data is not None,
        "samples_explained": min(arguments.max_samples, len(features)),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
