from hemzero.distillation.pysr_student import capability as pysr_capability
from hemzero.orchestration.taskweaver_adapter import verify_runtime


def test_pysr_runtime_is_configured():
    status = pysr_capability()
    assert status.available, status.reason
    assert "julia" in status.reason.lower()


def test_taskweaver_runtime_and_catalog():
    status = verify_runtime()
    assert status["available"], status["reason"]
    assert status["registered_tools"] == 5
