#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd
from dotenv import load_dotenv

from hemzero.distillation.pysr_student import capability as pysr_capability
from hemzero.models.tabpfn_views import create_tabpfn_classifier, tabpfn_capability
from hemzero.orchestration.taskweaver_adapter import verify_runtime


def main() -> None:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Verify HemoZero optional backends")
    parser.add_argument("--tabpfn-fit", action="store_true", help="run a real CPU fit; requires TABPFN_TOKEN")
    args = parser.parse_args()

    report: dict = {
        "pysr": pysr_capability().__dict__,
        "taskweaver": verify_runtime(),
        "tabpfn": tabpfn_capability().__dict__,
    }
    if args.tabpfn_fit:
        status = tabpfn_capability()
        if not status.authenticated:
            raise SystemExit("TABPFN_TOKEN is not set; accept the license and export it first")
        train = pd.read_csv("dataset/dataset/anxiety_train.csv")
        test = pd.read_csv("dataset/dataset/anxiety_test.csv")
        features = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "CORT"]
        model = create_tabpfn_classifier(device="cpu", n_estimators=1)
        model.set_params(
            auto_scale_n_estimators=False,
            fit_mode="low_memory",
            memory_saving_mode=True,
            keep_cache_on_device=False,
            n_preprocessing_jobs=1,
        )
        model.fit(train[features], train["Anxiety_14"])
        probability = model.predict_proba(test[features])[:, 1]
        report["tabpfn_real_fit"] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "probability_min": float(probability.min()),
            "probability_max": float(probability.max()),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
