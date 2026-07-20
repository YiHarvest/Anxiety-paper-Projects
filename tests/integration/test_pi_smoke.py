from pathlib import Path

from pi_agent.runner import PiRunner


def test_pi_pipeline_to_splits(tmp_path):
    project = Path(__file__).resolve().parents[2]
    runner = PiRunner(project)
    runner.experiment = {**runner.experiment, "artifact_root": str(tmp_path / "runs"), "tracking_uri": str(tmp_path / "mlruns")}
    state = runner.start(run_id="integration")
    state = runner.resume(Path(state.run_dir), until="create_cv_splits")
    assert state.completed == ["inspect_dataset", "validate_ratios", "build_lineage_graph", "create_cv_splits"]
    assert (Path(state.run_dir) / "splits" / "outer_split_assignment.csv").exists()
