from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hemzero.common.io import load_yaml


@dataclass(frozen=True)
class ToolContext:
    project_root: Path
    run_dir: Path

    def config(self, name: str) -> dict:
        return load_yaml(self.project_root / "configs" / name)


def require_file(value, name: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"{name} must be an existing file: {path}")
    return path
