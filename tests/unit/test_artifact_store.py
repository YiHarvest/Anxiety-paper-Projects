import json

import pytest

from hemzero.reporting.artifacts import ArtifactStore
from hemzero.reporting.evidence import verify_manifest


def test_artifact_store_freezes_and_verifies(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    run = store.create_run("test-run")
    (run / "reports" / "value.txt").write_text("evidence", encoding="utf-8")
    manifest_path = store.finalize(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_manifest(run, manifest) == []
    with pytest.raises(PermissionError):
        store.assert_writable(run)


def test_run_id_cannot_be_reused(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    store.create_run("same")
    with pytest.raises(FileExistsError):
        store.create_run("same")
