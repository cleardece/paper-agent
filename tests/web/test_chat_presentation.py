from web.app import ChatMessage, Session, serialize_session, timeline_snapshot


def test_timeline_preserves_agent_duration():
    timeline = timeline_snapshot([
        {"agent": "retriever", "status": "completed", "duration_ms": 83.4}
    ])

    step = next(item for item in timeline if item["agent"] == "retriever")
    assert step["duration_ms"] == 83.4


def test_session_serialization_includes_evidence_report():
    session = Session(id="s1", title="test")
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="answer",
            evidence_report={"status": "pass"},
        )
    )

    payload = serialize_session(session, include_messages=True)

    assert payload["messages"][0]["evidence_report"]["status"] == "pass"
