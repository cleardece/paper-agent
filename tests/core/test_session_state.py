from core.paper_context import PaperContext, PaperFocusState
from core.session_state import SessionStateReducer, normalize_agent_result


def context(primary="P001", paper_ids=None, status="resolved"):
    return PaperContext(
        primary_paper_id=primary,
        paper_ids=list(paper_ids or ([primary] if primary else [])),
        status=status,
        source="session_focus",
        confidence=0.98,
        inherited=True,
    )


def test_successful_result_updates_primary_focus():
    focus = PaperFocusState()

    updated = SessionStateReducer().reduce(
        focus,
        {
            "error": None,
            "primary_paper_id": "P003",
            "resolved_paper_ids": ["P003"],
        },
        context(primary="P003"),
    )

    assert updated.primary_paper_id == "P003"
    assert updated.active_paper_ids == ["P003"]
    assert updated.last_resolved_at is not None


def test_failed_result_does_not_overwrite_existing_focus():
    focus = PaperFocusState(
        primary_paper_id="P001",
        active_paper_ids=["P001"],
    )

    updated = SessionStateReducer().reduce(
        focus,
        {
            "error": "NEED_PAPER_CONTEXT",
            "primary_paper_id": None,
            "resolved_paper_ids": [],
        },
        context(primary=None, status="unresolved"),
    )

    assert updated.primary_paper_id == "P001"
    assert updated.active_paper_ids == ["P001"]


def test_comparison_keeps_primary_and_all_active_papers():
    focus = PaperFocusState(
        primary_paper_id="P001",
        active_paper_ids=["P001"],
    )

    updated = SessionStateReducer().reduce(
        focus,
        {
            "error": None,
            "primary_paper_id": "P001",
            "resolved_paper_ids": ["P001", "P002"],
        },
        context(primary="P001", paper_ids=["P001", "P002"]),
    )

    assert updated.primary_paper_id == "P001"
    assert updated.active_paper_ids == ["P001", "P002"]


def test_legacy_resolved_paper_id_is_normalized():
    result = normalize_agent_result({
        "error": None,
        "resolved_paper_id": "P001",
    })

    assert result["primary_paper_id"] == "P001"
    assert result["resolved_paper_ids"] == ["P001"]
