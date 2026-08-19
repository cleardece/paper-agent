# Research Conversation Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a research conversation's active paper and task across turns while silently building a correctable, source-traceable research profile.

**Architecture:** Persist deterministic session focus in the existing `sessions` document and pass it into `AgentState`; Supervisor resolves follow-up questions from that stable ID before consulting an LLM. Add a small `ResearchMemoryService` over MongoDB for structured profile snapshots and append-only change events; it extracts only user-provided profile updates after an answer and never supplies scientific evidence to the answer pipeline.

**Tech Stack:** Python 3.12, FastAPI/SSE, LangGraph, MongoDB/PyMongo, LangChain chat model, pytest.

---

## File map

- Modify: `state/graph_state.py` — declare typed session-focus and research-context state.
- Modify: `web/app.py` — persist and load focus/summary, build bounded context, schedule silent profile updates, expose profile read endpoint.
- Modify: `storage/mongodb.py` — preserve additional session fields without resetting `created_at`.
- Create: `storage/research_memory.py` — structured profile snapshots, audit events, extraction and merge rules.
- Modify: `core/deps.py` — construct and inject the research-memory service once.
- Modify: `agents/supervisor.py` — resolve follow-ups with `active_paper_ids`, remove unsafe title-regex selection.
- Modify: `agents/direct_analyzer.py` — return the paper ID actually analyzed so the web layer can make it the session focus.
- Create: `tests/storage/test_research_memory.py` — unit-test merge, override, provenance and extractor failures.
- Modify: `tests/web/test_paper_selection.py` — test session persistence and context construction.
- Create: `tests/agents/test_supervisor_session_focus.py` — test follow-up, switch, stale-focus and no-guess routing.
- Modify: `README.md` — document the evidence boundary and user correction behavior.

### Task 1: Persist an explicit, bounded session focus

**Files:**
- Modify: `web/app.py:58-151, 301-333, 901-979`
- Modify: `storage/mongodb.py:148-162`
- Modify: `state/graph_state.py:46-58`
- Modify: `tests/web/test_paper_selection.py`

- [ ] **Step 1: Write failing persistence tests**

```python
from web.app import Session, create_web_initial_state


def test_session_focus_survives_initial_state_construction():
    session = Session(id="s1", title="论文", active_paper_ids=["paper-a"], active_task="复现实验")
    state = create_web_initial_state("它怎么复现", session)
    assert state["active_paper_ids"] == ["paper-a"]
    assert state["active_task"] == "复现实验"


def test_explicit_selection_replaces_session_focus():
    session = Session(id="s1", title="论文", active_paper_ids=["paper-a"])
    state = create_web_initial_state("分析", session, target_paper_id="paper-b")
    assert state["target_paper_id"] == "paper-b"
    assert state["active_paper_ids"] == ["paper-a"]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `pytest tests/web/test_paper_selection.py -q`

Expected: failure because `Session` does not accept `active_paper_ids` and the state has no focus fields.

- [ ] **Step 3: Add explicit fields and lossless persistence**

Add to `Session`:

```python
user_id: str = "local-user"
active_paper_ids: list[str] = field(default_factory=list)
active_section: str | None = None
active_task: str | None = None
open_questions: list[str] = field(default_factory=list)
conversation_summary: str = ""
summary_through_message_count: int = 0
```

Add `id: str = field(default_factory=lambda: uuid4().hex)` to `ChatMessage`, persist it beside each message, and generate a backwards-compatible ID when loading an older message. Load every session field with backwards-compatible defaults. Add the same values to `save_session_to_db`, and change `MongoDBClient.save_session` to accept `**session_fields` and use:

```python
update = {
    "$set": {"title": title, "updated_at": updated_at or datetime.now(timezone.utc).timestamp(),
             "messages": messages or [], **session_fields},
    "$setOnInsert": {"session_id": session_id, "created_at": datetime.now(timezone.utc)},
}
self.db["sessions"].update_one({"session_id": session_id}, update, upsert=True)
```

Extend `AgentState` and `create_web_initial_state` with `active_paper_ids`, `active_section`, `active_task`, `open_questions`, `conversation_summary`, and `user_id`. Do not change a stored focus in `create_web_initial_state`; the explicit selection is applied only after the workflow has resolved it.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/web/test_paper_selection.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the persistence unit**

```bash
git add state/graph_state.py web/app.py storage/mongodb.py tests/web/test_paper_selection.py
git commit -m "feat: persist research session focus"
```

### Task 2: Resolve follow-ups from stable focus, never a guessed title

**Files:**
- Modify: `agents/supervisor.py:59-196`
- Modify: `agents/direct_analyzer.py:31-110`
- Modify: `web/app.py:912-979`
- Create: `tests/agents/test_supervisor_session_focus.py`

- [ ] **Step 1: Write failing Supervisor tests**

```python
from agents.supervisor import SupervisorAgent


def test_followup_uses_active_paper_id_without_llm():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({"user_query": "我怎么复现它的实验", "active_paper_ids": ["mmc"]})
    assert result["next_agent"] == "direct"
    assert result["target_paper_id"] == "mmc"


def test_followup_without_focus_does_not_guess_title_or_search_arxiv():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({"user_query": "我怎么复现它的实验", "conversation_context": "用户: Matlab"})
    assert result["next_agent"] == "retriever"
    assert result["target_paper"] is None
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/agents/test_supervisor_session_focus.py -q`

Expected: the first test calls the LLM path and the second can produce a guessed title.

- [ ] **Step 3: Implement deterministic follow-up routing**

At the start of `SupervisorAgent.invoke`, preserve the existing explicit `target_paper_id` branch and add:

```python
active_paper_ids = state.get("active_paper_ids") or []
if self._is_followup_query(query) and len(active_paper_ids) == 1:
    return {"next_agent": "direct", "search_query": query, "error": None,
            "target_paper": None, "target_paper_id": active_paper_ids[0]}
```

Delete `_extract_paper_from_context` and all uses of its broad capitalized-English regex. For a follow-up with zero or multiple active papers, return `retriever` with no `target_paper` rather than converting Chinese follow-up text into an arXiv query. Retain LLM routing for explicit titles, comparisons and non-follow-up requests.

Make successful `DirectAnalyzerAgent.invoke` include `resolved_paper_id` in its result whenever it analyzes an existing library paper. In `event_stream`, capture this ID from a node output and, after the workflow ends, set `session.active_paper_ids = [resolved_paper_id]`, `session.active_task = request.message[:500]`, and append the query to `open_questions` only if it is not already the newest entry; retain at most ten open questions. If the selected paper no longer exists, clear only that stale ID and preserve the error response.

- [ ] **Step 4: Run Supervisor and direct-analyzer regression tests**

Run: `pytest tests/agents/test_supervisor_session_focus.py tests/agents/test_direct_analyzer_selection.py -q`

Expected: all tests pass; no test relies on the deleted regex helper.

- [ ] **Step 5: Commit the routing unit**

```bash
git add agents/supervisor.py agents/direct_analyzer.py web/app.py tests/agents/test_supervisor_session_focus.py tests/agents/test_direct_analyzer_selection.py
git commit -m "feat: resolve research follow-ups from session focus"
```

### Task 3: Build bounded conversation context

**Files:**
- Create: `core/conversation_context.py`
- Modify: `web/app.py:301-333, 912-979`
- Modify: `tests/web/test_paper_selection.py`

- [ ] **Step 1: Write failing context-builder tests**

```python
from core.conversation_context import build_conversation_context


def test_context_keeps_summary_then_recent_messages():
    messages = [{"role": "user", "content": f"m{n}"} for n in range(12)]
    result = build_conversation_context("早期摘要", messages, recent_message_limit=4)
    assert result.startswith("早期摘要")
    assert "m8" in result and "m11" in result
    assert "m0" not in result
```

- [ ] **Step 2: Run the new test and confirm it fails**

Run: `pytest tests/web/test_paper_selection.py -q`

Expected: import error because `core.conversation_context` does not exist.

- [ ] **Step 3: Implement deterministic context composition and asynchronous summary refresh**

Create `build_conversation_context(summary, messages, recent_message_limit=10)` that formats `用户:` / `助手:` lines, always excludes the just-added current user message, and caps each message at 500 characters. Create `needs_summary_refresh(message_count, summary_through_message_count)` that returns true after ten unsummarized messages.

Add `refresh_summary(llm, previous_summary, messages)` with an explicit prompt that only summarizes conversation state, open questions and user-stated research progress; on any LLM or JSON failure return `previous_summary`. In `event_stream`, after saving the answer, schedule this refresh with `asyncio.create_task`; it updates only `conversation_summary` and `summary_through_message_count` and logs failures. `create_web_initial_state` must use `build_conversation_context` instead of slicing messages directly.

- [ ] **Step 4: Run context tests**

Run: `pytest tests/web/test_paper_selection.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the context unit**

```bash
git add core/conversation_context.py web/app.py tests/web/test_paper_selection.py
git commit -m "feat: bound research conversation context"
```

### Task 4: Store a source-traceable research profile and apply corrections

**Files:**
- Create: `storage/research_memory.py`
- Modify: `core/deps.py:67-147`
- Modify: `web/app.py:912-979`
- Create: `tests/storage/test_research_memory.py`

- [ ] **Step 1: Write failing research-memory tests**

```python
from storage.research_memory import ResearchMemoryService


def test_explicit_direction_correction_supersedes_prior_value(fake_db):
    service = ResearchMemoryService(fake_db, llm=None)
    service.apply_updates("local-user", [{"field": "research_directions", "value": ["computer vision"], "mode": "set", "confidence": 0.95}], "s1", ["m1"])
    profile = service.apply_updates("local-user", [{"field": "research_directions", "value": ["topology optimization"], "mode": "set", "confidence": 1.0, "explicit": True}], "s2", ["m2"])
    assert profile["research_directions"] == ["topology optimization"]
    assert service.events.count_documents({"user_id": "local-user"}) == 2


def test_extraction_failure_returns_no_updates(fake_db):
    service = ResearchMemoryService(fake_db, llm=FailingLLM())
    assert service.extract_updates("我正在复现实验", "", {}) == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/storage/test_research_memory.py -q`

Expected: import error because the service does not exist.

- [ ] **Step 3: Implement structured snapshots, audit events and safe extraction**

Create `ResearchMemoryService` with `profiles = db["research_profiles"]` and `events = db["research_memory_events"]`. Ensure a unique `user_id` index for profiles and indexes on event `user_id`, `updated_at` and `source_session_id`.

Use this profile schema:

```python
{
  "user_id": "local-user",
  "learning_status": [], "research_directions": [], "projects": [],
  "preferences": [], "updated_at": datetime,
}
```

Implement `extract_updates(user_message, conversation_summary, session_focus)` to ask the LLM for a JSON list containing only `learning_status`, `research_directions`, `projects`, or `preferences`. The prompt must state that inferred scientific claims and assistant-produced content are excluded. Parse only a JSON list; return `[]` on error. A no-LLM fallback recognizes no updates except explicit commands beginning with `记住`、`改为`、`删除` or `不是`.

Implement `apply_updates(user_id, updates, source_session_id, source_message_ids)`: validate fields, write an append-only event with old and new values, then update the profile. An explicit `mode="set"` replaces the field. Non-explicit updates append only values not already present. Store source IDs, confidence, `status="active"`, and `updated_at` in every event. Expose `get_profile(user_id)`.

Instantiate the service in `ServiceContainer` as `self.research_memory`.

- [ ] **Step 4: Schedule silent updates after a successful chat answer**

Add a local `async def update_research_memory()` inside `event_stream` that calls `container.research_memory.extract_updates` with the current user message, session summary and focus, then `apply_updates` with the session user ID and the newest user message's persisted ID. Start it with `asyncio.create_task` only after the answer and session have been persisted. Catch and log exceptions without changing SSE output or answer status.

- [ ] **Step 5: Run storage tests**

Run: `pytest tests/storage/test_research_memory.py -q`

Expected: all tests pass, including the LLM-failure non-blocking case.

- [ ] **Step 6: Commit the research-profile unit**

```bash
git add storage/research_memory.py core/deps.py web/app.py tests/storage/test_research_memory.py
git commit -m "feat: record source-traceable research profiles"
```

### Task 5: Make profile recall observable and protect the evidence boundary

**Files:**
- Modify: `web/app.py`
- Modify: `README.md`
- Modify: `tests/web/test_paper_selection.py`

- [ ] **Step 1: Write failing endpoint and boundary tests**

```python
def test_research_profile_endpoint_returns_profile(client, fake_container):
    fake_container.research_memory.get_profile.return_value = {"user_id": "local-user", "research_directions": ["PINN"]}
    response = client.get("/api/research-profile/local-user")
    assert response.status_code == 200
    assert response.json()["research_directions"] == ["PINN"]


def test_initial_state_keeps_profile_out_of_evidence_fields():
    state = create_web_initial_state("论文实验是什么")
    assert "profile_evidence" not in state
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/web/test_paper_selection.py -q`

Expected: the profile endpoint returns 404.

- [ ] **Step 3: Implement profile read endpoint and documentation**

Add `GET /api/research-profile/{user_id}`. It returns only the structured profile; audit events stay internal. Do not add profile content to `retrieved_chunks`, `evidence_report`, or any citation-building prompt. Update README with: what is remembered, silent update behavior, correction examples (`记住…`, `研究方向改为…`, `不是这样…`), and the rule that profile data never proves paper claims.

- [ ] **Step 4: Run endpoint and full test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Run import and whitespace verification**

Run: `python -c "from web.app import app; print('route count:', len(app.routes))"`

Expected: command exits 0 and prints a positive route count.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 6: Commit the API and documentation unit**

```bash
git add web/app.py README.md tests/web/test_paper_selection.py
git commit -m "feat: expose research profile safely"
```
