from pi_agent.policy_engine import PolicyEngine
from hemzero.knowledge_graph.models import InteractionAssessment, InteractionStatus


def test_policy_selects_only_unlocked_stage():
    policy = PolicyEngine({"stages": {"a": {"tool": "a", "requires": []}, "b": {"tool": "b", "requires": ["a"]}}})
    assert policy.next_stage([], {}, {}) == "a"
    assert policy.next_stage(["a"], {}, {}) == "b"
    assert policy.next_stage([], {}, {"a": "failed"}) is None


def test_policy_engine_is_final_interaction_decision_authority():
    assessment = InteractionAssessment(False, False, 0.8, 0.8, True, False)
    status = PolicyEngine.decide_interaction(
        assessment, {"min_frequency": 0.6, "min_direction_consistency": 0.6}
    )
    assert status == InteractionStatus.RETAIN_SUPPORTED
