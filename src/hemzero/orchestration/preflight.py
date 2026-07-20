"""Read-only backend, dataset and leakage preflight checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from hemzero.common.io import load_yaml
from hemzero.distillation.pysr_student import capability as pysr_capability
from hemzero.distillation.tabdistill_runner import capability as tabdistill_capability
from hemzero.models.tabpfn_views import tabpfn_capability


def model_preflight(root: Path) -> dict:
    load_dotenv(root / ".env", override=False)
    dataset = load_yaml(root / "configs" / "dataset.yaml")
    experiment = load_yaml(root / "configs" / "experiment.yaml")
    models = load_yaml(root / "configs" / "models.yaml")
    train_path, test_path = root / dataset["train_path"], root / dataset["test_path"]
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = dataset["raw_features"] + dataset["ratio_features"]
    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    shared = set(train_hash) & set(test_hash)
    tabpfn = tabpfn_capability()
    pysr = pysr_capability()
    tabdistill = tabdistill_capability(Path(models["tabdistill"]["repository"]))
    checks = {
        "train_exists": train_path.is_file(), "test_exists": test_path.is_file(),
        "train_rows": len(train), "test_rows": len(test),
        "patient_id_overlap": len(set(train[dataset["id_column"]]) & set(test[dataset["id_column"]])),
        "exact_feature_vector_overlap": int(test_hash.isin(shared).sum()),
        "tabpfn_available": tabpfn.available, "tabpfn_authenticated": tabpfn.authenticated,
        "pysr_available": pysr.available, "tabdistill_available": tabdistill.available,
        "execution_profile": experiment.get("execution", {}).get("profile"),
        "outer_repeats": experiment.get("execution", {}).get("outer_repeats"),
        "outer_folds": experiment["outer_splits"],
    }
    overlap_authorized = bool(dataset.get("allow_exact_feature_vector_overlap", False))
    checks["exact_feature_vector_overlap_authorized"] = overlap_authorized
    fatal = ["patient_id_overlap"] if checks["patient_id_overlap"] != 0 else []
    if checks["exact_feature_vector_overlap"] != 0 and not overlap_authorized:
        fatal.append("exact_feature_vector_overlap")
    fatal += [name for name in ("tabpfn_available", "tabpfn_authenticated", "pysr_available", "tabdistill_available") if not checks[name]]
    checks["pass"] = not fatal
    checks["fatal"] = fatal
    return checks
