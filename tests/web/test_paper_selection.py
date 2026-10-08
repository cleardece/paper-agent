import asyncio
from pathlib import Path

from core.conversation_context import build_conversation_context, needs_summary_refresh
from web.app import (
    ChatMessage,
    ChatRequest,
    Session,
    create_web_initial_state,
    get_research_profile,
    session_from_document,
    serialize_session,
    update_research_memory,
)


ROOT = Path(__file__).parents[2]


def test_chat_request_and_state_preserve_selected_paper_id():
    request = ChatRequest(message="分析论文", target_paper_id="local_paper_1")

    state = create_web_initial_state(
        request.message,
        target_paper_id=request.target_paper_id,
    )

    assert state["target_paper_id"] == "local_paper_1"


def test_library_and_chat_client_transmit_selected_paper_id():
    library = (ROOT / "web/static/papers.html").read_text(encoding="utf-8")
    chat = (ROOT / "web/static/app.js").read_text(encoding="utf-8")

    assert 'localStorage.setItem("pendingPaperId", arxivId)' in library
    assert "target_paper_id: targetPaperId" in chat


def test_session_focus_survives_initial_state_construction():
    session = Session(
        id="s1",
        title="论文",
        active_paper_ids=["paper-a"],
        active_task="复现实验",
    )

    state = create_web_initial_state("它怎么复现", session)

    assert state["active_paper_ids"] == ["paper-a"]
    assert state["active_task"] == "复现实验"
    assert state["paper_focus"]["primary_paper_id"] == "paper-a"


def test_explicit_selection_does_not_overwrite_stored_session_focus():
    session = Session(id="s1", title="论文", active_paper_ids=["paper-a"])

    state = create_web_initial_state("分析", session, target_paper_id="paper-b")

    assert state["target_paper_id"] == "paper-b"
    assert state["active_paper_ids"] == ["paper-a"]


def test_legacy_active_list_loads_as_primary_focus():
    session = session_from_document({
        "session_id": "s1",
        "title": "论文",
        "active_paper_ids": ["P001", "P002"],
        "messages": [],
    })

    assert session.paper_focus.primary_paper_id == "P001"
    assert session.paper_focus.active_paper_ids == ["P001", "P002"]


def test_recent_structured_paper_metadata_enters_initial_state():
    session = Session(id="s1", title="论文", active_paper_ids=["P001"])
    metadata = {
        "primary_paper_id": "P001",
        "paper_ids": ["P001"],
        "status": "resolved",
        "source": "session_focus",
        "confidence": 0.98,
        "inherited": True,
        "switched_from_paper_id": None,
    }
    session.messages.extend([
        ChatMessage(role="assistant", content="分析完成", paper_context=metadata),
        ChatMessage(role="user", content="总结实验结果"),
    ])

    state = create_web_initial_state("总结实验结果", session)

    assert state["recent_paper_contexts"] == [metadata]


def test_initial_state_exposes_only_prior_user_messages_to_query_rewriter():
    session = Session(id="s1", title="论文", active_paper_ids=["P001"])
    session.messages.extend([
        ChatMessage(role="user", content="它的方法有什么创新？"),
        ChatMessage(role="assistant", content="方法分析结果"),
        ChatMessage(role="user", content="实验呢？"),
    ])

    state = create_web_initial_state("实验呢？", session)

    assert state["recent_user_messages"] == ["它的方法有什么创新？"]
    assert state["retrieval_query"] is None


def test_session_serialization_preserves_paper_focus_and_message_metadata():
    metadata = {
        "primary_paper_id": "P002",
        "paper_ids": ["P002"],
        "status": "switch",
        "source": "explicit_target",
        "confidence": 1.0,
        "inherited": False,
        "switched_from_paper_id": "P001",
    }
    session = Session(id="s1", title="论文", active_paper_ids=["P002"])
    session.messages.append(
        ChatMessage(role="user", content="再看看 P002", paper_context=metadata)
    )

    payload = serialize_session(session, include_messages=True)

    assert payload["paper_focus"]["primary_paper_id"] == "P002"
    assert payload["messages"][0]["paper_context"] == metadata


def test_context_keeps_summary_then_only_recent_messages():
    messages = [{"role": "user", "content": f"m{index}"} for index in range(12)]

    result = build_conversation_context("早期摘要", messages, recent_message_limit=4)

    assert "早期摘要" in result
    assert "m8" in result and "m11" in result
    assert "m0" not in result


def test_summary_refresh_waits_for_ten_unsummarized_messages():
    assert not needs_summary_refresh(19, 10)
    assert needs_summary_refresh(20, 10)


def test_research_profile_update_is_silent_and_source_traceable(monkeypatch):
    class RecordingResearchMemory:
        def __init__(self):
            self.applied = None

        def extract_updates(self, *_args):
            return [{"field": "research_directions", "value": ["PINN"]}]

        def apply_updates(self, *args):
            self.applied = args

    memory = RecordingResearchMemory()
    container = type("Container", (), {"research_memory": memory})()
    monkeypatch.setattr("core.deps.get_container", lambda: container)
    session = Session(id="s1", title="论文")
    message = ChatMessage(role="user", content="我研究 PINN")

    asyncio.run(update_research_memory(session, message))

    assert memory.applied[0] == "local-user"
    assert memory.applied[3] == [message.id]


def test_research_profile_endpoint_returns_profile_without_evidence(monkeypatch):
    profile = {"user_id": "local-user", "research_directions": ["PINN"]}
    memory = type("ResearchMemory", (), {"get_profile": lambda self, _user_id: profile})()
    container = type("Container", (), {"research_memory": memory})()
    monkeypatch.setattr("core.deps.get_container", lambda: container)

    result = asyncio.run(get_research_profile("local-user"))

    assert result == profile
    assert "profile_evidence" not in create_web_initial_state("论文实验是什么")
