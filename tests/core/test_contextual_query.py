from core.contextual_query import ContextualQueryRewriter


class FakeResponse:
    def __init__(self, content):
        self.content = content


class RecordingLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return FakeResponse(self.response)


class FakePaperRepository:
    def get_paper(self, paper_id):
        if paper_id == "P001":
            return {"arxiv_id": paper_id, "title": "Paper A"}
        return None


def contextual_state():
    return {
        "user_query": "实验呢？",
        "intent": "analyze",
        "turn_context": {
            "intent": "analyze",
            "primary_paper_id": "P001",
            "paper_ids": ["P001"],
        },
        "active_section": "method",
        "active_task": "分析方法的有效性",
        "recent_user_messages": ["它的方法有什么创新？"],
        "conversation_summary": "用户正在分析 Paper A 的方法。",
    }


def test_rewrites_follow_up_with_bounded_context_and_paper_title():
    llm = RecordingLLM(
        '{"retrieval_query":"Paper A 中用于验证方法有效性的实验设置、数据集、指标和结果是什么？"}'
    )
    result = ContextualQueryRewriter(llm, FakePaperRepository()).invoke(
        contextual_state()
    )

    assert result["contextual_query_rewritten"] is True
    assert result["retrieval_query"].startswith("Paper A 中")
    prompt = llm.calls[0][-1].content
    assert "Paper A" in prompt
    assert "它的方法有什么创新" in prompt
    assert "只用于理解意图" in prompt


def test_without_prior_conversation_returns_original_query_without_llm_call():
    llm = RecordingLLM('{"retrieval_query":"unused"}')
    state = {
        "user_query": "P001 的实验设置和结果是什么？",
        "intent": "analyze",
        "turn_context": {"intent": "analyze", "paper_ids": ["P001"]},
    }

    result = ContextualQueryRewriter(llm, FakePaperRepository()).invoke(state)

    assert result == {
        "retrieval_query": state["user_query"],
        "contextual_query_rewritten": False,
    }
    assert llm.calls == []


def test_llm_failure_falls_back_without_setting_workflow_error():
    llm = RecordingLLM(error=RuntimeError("insufficient_quota"))

    result = ContextualQueryRewriter(llm, FakePaperRepository()).invoke(
        contextual_state()
    )

    assert result["retrieval_query"] == "实验呢？"
    assert result["contextual_query_rewritten"] is False
    assert "error" not in result
    assert len(llm.calls) == 1
