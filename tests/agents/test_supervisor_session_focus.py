from agents.supervisor import SupervisorAgent


def paper_context(primary_id=None, paper_ids=None):
    return {
        "primary_paper_id": primary_id,
        "paper_ids": list(paper_ids or ([primary_id] if primary_id else [])),
        "status": "resolved" if primary_id else "unresolved",
        "source": "session_focus" if primary_id else "none",
        "confidence": 0.98 if primary_id else 0.0,
        "inherited": bool(primary_id),
    }


def test_non_pronominal_structured_followup_is_analyze_without_llm():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "请从方法、实验设置、消融和局限四方面给出结构化分析",
        "paper_context": paper_context("P001"),
    })
    assert result["intent"] == "analyze"
    assert "target_paper_id" not in result
    assert "search_query" not in result


def test_short_followup_with_focus_is_analyze_without_pronoun_requirement():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "总结实验结果",
        "paper_context": paper_context("P001"),
    })
    assert result["intent"] == "analyze"


def test_find_similar_papers_is_search_even_with_current_focus():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "帮我找几篇类似论文",
        "paper_context": paper_context("P001"),
    })
    assert result["intent"] == "search"


def test_english_find_similar_papers_is_search():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "find similar papers about this method",
        "paper_context": paper_context("P001"),
    })
    assert result["intent"] == "search"


def test_compare_intent_is_independent_from_paper_resolution():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "把这两篇论文对比一下",
        "paper_context": paper_context("P001", ["P001", "P002"]),
    })
    assert result["intent"] == "compare"


def test_unresolved_anaphoric_analysis_remains_analyze_for_policy_gate():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "分析它的实验方法",
        "paper_context": paper_context(),
    })
    assert result["intent"] == "analyze"


def test_greeting_is_general():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "你好",
        "paper_context": paper_context(),
    })
    assert result["intent"] == "general"
