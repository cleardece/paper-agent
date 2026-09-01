from agents.presenter import PresenterAgent
from agents.reflector import ReflectorAgent
from core.llm_errors import (
    LLM_QUOTA_EXHAUSTED,
    LLM_RATE_LIMITED,
    agent_failure_result,
    classify_llm_error,
)


class RecordingFailingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        raise AssertionError("LLM must not be called after an upstream error")

    def stream(self, _messages):
        self.calls += 1
        raise AssertionError("LLM must not be called after an upstream error")


def test_provider_quota_errors_are_classified_without_another_llm_call():
    assert classify_llm_error("Allocated quota exceeded; insufficient_quota") == LLM_QUOTA_EXHAUSTED
    assert classify_llm_error("rpm exhausted") == LLM_RATE_LIMITED

    result = agent_failure_result(
        "direct_analyzer",
        RuntimeError("Allocated quota exceeded; insufficient_quota"),
    )

    assert result["error"] == LLM_QUOTA_EXHAUSTED
    assert result["error_detail"] == "Allocated quota exceeded; insufficient_quota"
    assert "配额" in result["answer"]


def test_rate_limit_is_not_misclassified_as_permanent_quota_exhaustion():
    result = agent_failure_result("presenter", RuntimeError("rpm exhausted"))

    assert result["error"] == LLM_RATE_LIMITED
    assert "稍后重试" in result["answer"]


def test_presenter_does_not_call_llm_after_upstream_quota_error():
    llm = RecordingFailingLLM()
    presenter = PresenterAgent(llm)

    result = presenter.invoke({
        "user_query": "Agent 能控制什么？",
        "error": "direct_analyzer 执行失败: insufficient_quota",
        "analysis": None,
        "answer": None,
        "retrieved_chunks": [],
    })

    assert llm.calls == 0
    assert "配额" in result["answer"]


def test_reflector_does_not_call_llm_after_upstream_error():
    llm = RecordingFailingLLM()
    reflector = ReflectorAgent(llm)

    result = reflector.invoke({
        "error": "LLM_QUOTA_EXHAUSTED",
        "analysis": "已有分析",
        "retrieved_chunks": [],
        "target_papers": [],
    })

    assert llm.calls == 0
    assert result["reflection"] is None
