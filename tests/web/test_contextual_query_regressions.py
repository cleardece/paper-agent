from agents.direct_analyzer import DirectAnalyzerAgent
from agents.retriever import RetrieverAgent
from core.contextual_query import ContextualQueryRewriter
from web.app import ChatMessage, Session, create_web_initial_state


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


class FakeUserMemory:
    def __init__(self):
        self.interactions = []

    def get_interests(self, _user_id, top_k=5):
        return []

    def process_interaction(self, _user_id, query):
        self.interactions.append(query)


class FakeMongo:
    def __init__(self):
        self.user_memory = FakeUserMemory()

    def get_paper(self, paper_id):
        if paper_id == "P001":
            return {"arxiv_id": paper_id, "title": "Paper A", "status": "indexed"}
        return None

    def get_chunks_by_paper(self, paper_id):
        if paper_id == "P001":
            return [{
                "paper_arxiv_id": paper_id,
                "chunk_index": 1,
                "content": "paper evidence",
                "metadata": {"section": "Experiment"},
            }]
        return []


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


def test_contextual_follow_up_becomes_standalone_query_without_changing_scope():
    llm = RecordingLLM(
        '{"retrieval_query":"Paper A 中用于验证方法有效性的实验设置、数据集、指标和结果是什么？"}'
    )
    state = contextual_state()

    result = ContextualQueryRewriter(llm, FakeMongo()).invoke(state)

    assert result["contextual_query_rewritten"] is True
    assert result["retrieval_query"].startswith("Paper A 中")
    assert state["turn_context"]["paper_ids"] == ["P001"]
    prompt = llm.calls[0][-1].content
    assert "它的方法有什么创新" in prompt
    assert "只用于理解意图" in prompt


def test_rewriter_can_resolve_reference_to_previous_assistant_answer():
    llm = RecordingLLM(
        '{"retrieval_query":"Paper A 的三个核心发现分别是什么，它们之间如何整合？"}'
    )
    state = contextual_state()
    state["user_query"] = "把你刚才的三个答案整合一下"
    state["conversation_context"] = (
        "用户: 请回答贡献、实验效果和控制代价三个问题\n"
        "助手: 答案一是首次使用PPO；答案二是减阻约8%；"
        "答案三是射流质量流量约0.5%。"
    )

    result = ContextualQueryRewriter(llm, FakeMongo()).invoke(state)

    assert result["contextual_query_rewritten"] is True
    prompt = llm.calls[0][-1].content
    assert "答案一是首次使用PPO" in prompt
    assert "助手回答只能用于识别追问对象" in prompt


def test_passthrough_and_quota_fallback_do_not_fail_workflow():
    no_history = {
        "user_query": "P001 的实验设置和结果是什么？",
        "intent": "analyze",
        "turn_context": {"intent": "analyze", "paper_ids": ["P001"]},
    }
    unused_llm = RecordingLLM('{"retrieval_query":"unused"}')
    assert ContextualQueryRewriter(unused_llm, FakeMongo()).invoke(no_history) == {
        "retrieval_query": no_history["user_query"],
        "contextual_query_rewritten": False,
    }
    assert unused_llm.calls == []

    failing_llm = RecordingLLM(error=RuntimeError("insufficient_quota"))
    assert ContextualQueryRewriter(failing_llm, FakeMongo()).invoke(
        contextual_state()
    ) == {
        "retrieval_query": "实验呢？",
        "contextual_query_rewritten": False,
    }
    assert len(failing_llm.calls) == 1


def test_web_state_supplies_only_prior_user_messages():
    session = Session(id="s1", title="论文", active_paper_ids=["P001"])
    session.messages.extend([
        ChatMessage(role="user", content="它的方法有什么创新？"),
        ChatMessage(role="assistant", content="方法分析结果"),
        ChatMessage(role="user", content="实验呢？"),
    ])

    state = create_web_initial_state("实验呢？", session)

    assert state["recent_user_messages"] == ["它的方法有什么创新？"]
    assert state["retrieval_query"] is None


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [[1.0] for _ in texts]


class RecordingRetriever(RetrieverAgent):
    def __init__(self, embedder, mongo):
        super().__init__(embedder, object(), mongo, llm=None)
        self.multi_query_input = None
        self.section_query = None

    def _multi_query(self, query, session_id=None, user_interests=None):
        self.multi_query_input = query
        return [query]

    def _detect_section_intent(self, query):
        self.section_query = query
        return None

    def _two_level_retrieval(self, query, query_vector, session_id=None,
                             paper_ids=None, graph_paper_ids=None):
        return [{
            "paper_arxiv_id": "P001",
            "chunk_index": 1,
            "content": "paper evidence",
            "score": 1.0,
            "section": "Experiment",
        }]


def test_retriever_uses_rewrite_but_evidence_and_memory_stay_separate():
    embedder = FakeEmbedder()
    mongo = FakeMongo()
    retriever = RecordingRetriever(embedder, mongo)
    standalone = "Paper A 的实验设置、指标和结果是什么？"

    result = retriever.invoke({
        "user_query": "实验呢？",
        "retrieval_query": standalone,
        "user_id": "u1",
        "turn_context": {"primary_paper_id": "P001", "paper_ids": ["P001"]},
    })

    assert retriever.multi_query_input == standalone
    assert retriever.section_query == standalone
    assert embedder.texts == [standalone]
    assert mongo.user_memory.interactions == ["实验呢？"]
    assert result["retrieved_chunks"][0]["content"] == "paper evidence"


class RecordingDirectAnalyzer(DirectAnalyzerAgent):
    def __init__(self, mongo):
        super().__init__(None, mongo, None, None, None)
        self.analysis_query = None

    def _analyze(self, _paper_info, _core_text, query=""):
        self.analysis_query = query
        return {"answer": "analysis", "error": None}


def test_direct_analyzer_uses_the_same_standalone_query():
    agent = RecordingDirectAnalyzer(FakeMongo())
    standalone = "Paper A 的实验设置、指标和结果是什么？"

    agent.invoke({
        "user_query": "实验呢？",
        "retrieval_query": standalone,
        "target_paper_id": "P001",
    })

    assert agent.analysis_query == standalone
