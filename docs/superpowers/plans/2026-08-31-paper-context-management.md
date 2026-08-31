# Paper Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve paper identity before intent routing, enforce explicit external-search admission, and persist a stable primary paper across all successful paper-analysis paths.

**Architecture:** Add deterministic paper-context resolution before Supervisor and a turn-context policy step after it. Downstream agents consume the normalized turn context, arXiv calls pass through one guarded search interface, and a pure reducer applies normalized paper results to session focus and message metadata.

**Tech Stack:** Python 3.13, FastAPI/SSE, LangGraph, MongoDB/PyMongo, LangChain chat models, pytest.

---

## File map

- Create `core/paper_context.py` — paper focus/context models and deterministic resolver.
- Create `core/search_policy.py` — validated search requests, query builder, admission gate, guarded arXiv call.
- Create `core/turn_context.py` — normalized intent/turn context construction and routing invariants.
- Create `core/session_state.py` — normalized agent result helpers and the only session-focus reducer.
- Modify `state/graph_state.py` — declare paper context, turn context, normalized result, and compatibility projections.
- Modify `agents/supervisor.py` — classify intent only from query plus resolved context.
- Modify `agents/direct_analyzer.py` — require a resolved local target and remove arXiv fallback.
- Modify `agents/fetcher.py` — consume validated search requests through the admission gate and return paper IDs.
- Modify `agents/retriever.py` — consume resolved paper IDs and remove history/title guessing.
- Modify `agents/analyzer.py` — return normalized paper IDs from retrieved evidence.
- Modify `agents/presenter.py` — preserve normalized paper result fields when returning an upstream answer.
- Modify `core/deps.py`, `main.py` — construct the resolver/policy components and remove DirectAnalyzer's arXiv dependency.
- Modify `graph/workflow.py`, `web/app.py` — add resolver and turn-context nodes, persist focus/metadata, and reduce results once.
- Create `tests/core/test_paper_context.py` — resolver priority, inheritance, switch, comparison, history, ambiguity.
- Create `tests/core/test_search_policy.py` — builder and external-search admission invariants.
- Modify `tests/agents/test_supervisor_session_focus.py` — intent-only routing with non-pronominal follow-ups.
- Modify `tests/agents/test_direct_analyzer_selection.py` — required target and zero-search behavior.
- Create `tests/agents/test_fetcher_search_policy.py` — exact guarded arXiv calls and normalized IDs.
- Modify `tests/web/test_paper_selection.py` — persistence and message metadata compatibility.
- Create `tests/web/test_paper_context_regressions.py` — seven requested end-to-end regression cases.

### Task 1: Add paper focus and resolution as an independent domain boundary

**Files:**
- Create: `core/paper_context.py`
- Modify: `state/graph_state.py`
- Create: `tests/core/test_paper_context.py`

- [ ] **Step 1: Write failing resolver tests**

Create tests with a `FakePaperRepository` exposing `get_paper()` and `list_papers()`:

```python
def paper(paper_id: str, title: str, doi: str | None = None) -> dict:
    return {"arxiv_id": paper_id, "title": title, "doi": doi, "abstract": ""}


class FakePaperRepository:
    def __init__(self, papers: list[dict]):
        self.papers = papers

    def get_paper(self, paper_id: str):
        return next((item for item in self.papers if item["arxiv_id"] == paper_id), None)

    def list_papers(self, **_kwargs):
        return list(self.papers)


def test_session_primary_is_inherited_without_pronouns():
    resolver = PaperContextResolver(FakePaperRepository([paper("P001", "Paper One")]))
    context = resolver.resolve(
        query="总结实验结果",
        explicit_target_paper_id=None,
        paper_focus=PaperFocusState(primary_paper_id="P001", active_paper_ids=["P001"]),
        recent_paper_contexts=[],
    )
    assert context.primary_paper_id == "P001"
    assert context.source == "session_focus"
    assert context.inherited is True


def test_explicit_target_switches_primary():
    resolver = PaperContextResolver(FakePaperRepository([paper("P001", "One"), paper("P002", "Two")]))
    context = resolver.resolve(
        query="分析",
        explicit_target_paper_id="P002",
        paper_focus=PaperFocusState(primary_paper_id="P001", active_paper_ids=["P001"]),
        recent_paper_contexts=[],
    )
    assert context.status == "switch"
    assert context.primary_paper_id == "P002"
    assert context.switched_from_paper_id == "P001"


def test_compare_keeps_current_primary_and_adds_title_match():
    resolver = PaperContextResolver(FakePaperRepository([paper("P001", "One"), paper("P002", "Paper B")]))
    context = resolver.resolve(
        query="把当前论文和 Paper B 对比一下",
        explicit_target_paper_id=None,
        paper_focus=PaperFocusState(primary_paper_id="P001", active_paper_ids=["P001"]),
        recent_paper_contexts=[],
    )
    assert context.primary_paper_id == "P001"
    assert context.paper_ids == ["P001", "P002"]
```

Also cover arXiv IDs, DOI matches, history metadata fallback, explicit search not being mistaken for an analysis target, and multiple title matches returning `ambiguous`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/core/test_paper_context.py -q`

Expected: FAIL because `core.paper_context` does not exist.

- [ ] **Step 3: Implement models and deterministic resolver**

Define dataclasses with lossless `to_dict()` / `from_dict()` helpers:

```python
@dataclass
class PaperFocusState:
    primary_paper_id: str | None = None
    active_paper_ids: list[str] = field(default_factory=list)
    source: str | None = None
    confidence: float | None = None
    last_resolved_at: datetime | None = None


@dataclass(frozen=True)
class PaperContext:
    primary_paper_id: str | None
    paper_ids: list[str]
    status: str
    source: str
    confidence: float
    inherited: bool
    switched_from_paper_id: str | None = None
```

Implement `PaperContextResolver.resolve()` in the documented priority order. Normalize
versioned arXiv IDs, match DOI fields case-insensitively, match explicit/contained
normalized local titles, preserve insertion order while deduplicating IDs, and consult
only structured history metadata. `is_explicit_search_request()` prevents ordinary
discovery requests from inheriting the session paper as their analysis target, except
that “similar papers” may retain the current paper as search context.

Extend `AgentState` with `paper_focus`, `paper_context`, `recent_paper_contexts`,
`intent`, `turn_context`, `primary_paper_id`, and `resolved_paper_ids`. Keep legacy
`target_paper_id`, `resolved_paper_id`, and `active_paper_ids` during migration.

- [ ] **Step 4: Run resolver tests**

Run: `.venv\Scripts\python.exe -m pytest tests/core/test_paper_context.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the resolution boundary**

```powershell
git add core/paper_context.py state/graph_state.py tests/core/test_paper_context.py
git commit -m "feat: resolve paper context before routing"
```

### Task 2: Make Supervisor intent-only and build an invariant-checked TurnContext

**Files:**
- Create: `core/turn_context.py`
- Modify: `agents/supervisor.py`
- Modify: `tests/agents/test_supervisor_session_focus.py`
- Create: `tests/core/test_turn_context.py`

- [ ] **Step 1: Write failing intent and invariant tests**

```python
def resolved_context(paper_id: str) -> dict:
    return {
        "primary_paper_id": paper_id,
        "paper_ids": [paper_id],
        "status": "resolved",
        "source": "session_focus",
        "confidence": 0.98,
        "inherited": True,
        "switched_from_paper_id": None,
    }


def unresolved_context() -> dict:
    return {
        "primary_paper_id": None,
        "paper_ids": [],
        "status": "unresolved",
        "source": "none",
        "confidence": 0.0,
        "inherited": False,
        "switched_from_paper_id": None,
    }


def test_non_pronominal_followup_is_analyze_without_identity_guessing():
    agent = SupervisorAgent(llm=None)
    result = agent.invoke({
        "user_query": "请从方法、实验设置、消融和局限四方面结构化分析",
        "paper_context": resolved_context("P001"),
    })
    assert result["intent"] == "analyze"
    assert "target_paper_id" not in result


def test_analyze_without_primary_is_stopped_before_direct_analyzer():
    result = build_turn_context("分析它的实验方法", "analyze", unresolved_context())
    assert result["error"] == "NEED_PAPER_CONTEXT"
    assert result["next_agent"] == "END"


def test_only_search_intent_admits_external_search():
    analyze = build_turn_context("分析", "analyze", resolved_context("P001"))
    search = build_turn_context("帮我找几篇类似论文", "search", unresolved_context())
    assert analyze["turn_context"]["allow_external_search"] is False
    assert search["turn_context"]["allow_external_search"] is True
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_supervisor_session_focus.py tests/core/test_turn_context.py -q`

Expected: FAIL because Supervisor still resolves identity and no turn-context builder exists.

- [ ] **Step 3: Implement intent-only routing**

Replace the Supervisor prompt with intent-only JSON:

```json
{"intent": "analyze|rag|compare|search|general", "reason": "short routing rationale"}
```

Use deterministic high-confidence rules for explicit search, comparison, greetings,
and paper-scoped analysis; use the LLM only for remaining intent ambiguity. Pass the
serialized `paper_context` only as context. Delete `_is_followup_query()`,
`_has_explicit_paper_reference()`, `_check_paper_exists()`, title extraction, and every
Supervisor assignment to `target_paper_id`.

- [ ] **Step 4: Implement TurnContext construction and routing invariants**

Add:

```python
def build_turn_context(query: str, intent: str, paper_context: dict) -> dict:
    allow_search = intent == "search"
    primary_id = paper_context.get("primary_paper_id")
    error = "NEED_PAPER_CONTEXT" if intent == "analyze" and not primary_id else None
    next_agent = {
        "analyze": "direct",
        "rag": "retriever",
        "compare": "retriever",
        "search": "fetcher",
        "general": "END",
    }[intent]
    if error:
        next_agent = "END"
    return {
        "intent": intent,
        "turn_context": {
            "query": query,
            "intent": intent,
            "primary_paper_id": primary_id,
            "paper_ids": list(paper_context.get("paper_ids") or []),
            "paper_resolution_source": paper_context.get("source", "none"),
            "paper_resolution_confidence": paper_context.get("confidence", 0.0),
            "allow_external_search": allow_search,
            "search_request": None,
        },
        "next_agent": next_agent,
        "error": error,
    }
```

Expose an `invoke(state)` adapter so this function can be a LangGraph node. Populate
legacy `target_paper_id` only here as a compatibility projection from the already
resolved primary ID.

- [ ] **Step 5: Run intent and turn-context tests**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_supervisor_session_focus.py tests/core/test_turn_context.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the routing separation**

```powershell
git add core/turn_context.py agents/supervisor.py tests/agents/test_supervisor_session_focus.py tests/core/test_turn_context.py
git commit -m "feat: separate paper identity from intent routing"
```

### Task 3: Enforce validated and explicitly admitted arXiv search

**Files:**
- Create: `core/search_policy.py`
- Create: `tests/core/test_search_policy.py`

- [ ] **Step 1: Write failing search-policy tests**

```python
class RecordingArxiv:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        return self.results[:max_results]


def test_denied_turn_cannot_call_arxiv():
    arxiv = RecordingArxiv()
    with pytest.raises(ExternalSearchNotAllowed):
        guarded_arxiv_search(arxiv, analyze_turn("P001"), SearchRequest("keywords", "PINN"), 5)
    assert arxiv.calls == []


def test_builder_strips_analysis_requirements_from_search_query():
    raw = "帮我找 PINN 求解 Navier-Stokes 的论文，并分析以下八个问题：方法、实验、局限"
    request = SearchQueryBuilder().build(raw)
    assert request.mode == "keywords"
    assert "PINN" in request.value and "Navier-Stokes" in request.value
    assert request.value != raw
    assert "八个问题" not in request.value


def test_arxiv_id_builds_canonical_request():
    request = SearchQueryBuilder().build("搜索 arXiv:2401.01234v2")
    assert request == SearchRequest(mode="arxiv_id", value="2401.01234")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/core/test_search_policy.py -q`

Expected: FAIL because `core.search_policy` does not exist.

- [ ] **Step 3: Implement the query builder and admission gate**

Define immutable `SearchRequest`, `ExternalSearchNotAllowed`, and
`InvalidSearchRequest`. `SearchQueryBuilder.build(query, fallback_title=None)` must:

- canonicalize arXiv IDs;
- use explicitly quoted titles as `title` requests;
- trim the text after analysis/answer clauses;
- remove request boilerplate and retain meaningful Chinese terms plus academic Latin
  tokens;
- use the current paper title for “similar papers” when supplied;
- reject empty values and values over 200 characters.

Implement:

```python
def guarded_arxiv_search(arxiv, turn_context, request, max_results):
    if not turn_context.get("allow_external_search") or turn_context.get("intent") != "search":
        raise ExternalSearchNotAllowed("external search requires SEARCH intent")
    if not isinstance(request, SearchRequest):
        request = SearchRequest.from_dict(request)
    request.validate()
    return arxiv.search(request.value, max_results=max_results)
```

The turn-context node builds and serializes a request only for search intent. It uses
the local primary paper title as fallback context for “similar papers.”

- [ ] **Step 4: Run search-policy tests**

Run: `.venv\Scripts\python.exe -m pytest tests/core/test_search_policy.py tests/core/test_turn_context.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the search boundary**

```powershell
git add core/search_policy.py core/turn_context.py tests/core/test_search_policy.py tests/core/test_turn_context.py
git commit -m "feat: gate and validate external paper search"
```

### Task 4: Remove DirectAnalyzer fallback and normalize agent paper results

**Files:**
- Create: `core/session_state.py`
- Modify: `agents/direct_analyzer.py`
- Modify: `agents/fetcher.py`
- Modify: `agents/retriever.py`
- Modify: `agents/analyzer.py`
- Modify: `agents/presenter.py`
- Modify: `tests/agents/test_direct_analyzer_selection.py`
- Create: `tests/agents/test_fetcher_search_policy.py`
- Create: `tests/core/test_session_state.py`

- [ ] **Step 1: Write failing DirectAnalyzer, Fetcher, and reducer tests**

```python
def paper(paper_id: str, title: str) -> dict:
    return {
        "arxiv_id": paper_id,
        "title": title,
        "abstract": "",
        "authors": [],
        "pdf_url": "https://example.test/paper.pdf",
    }


def test_direct_analyzer_requires_resolved_target_and_never_searches():
    agent = RecordingDirectAnalyzer(FakeMongo(None, []))
    result = agent.invoke({"user_query": "分析它的实验方法", "turn_context": analyze_turn(None)})
    assert result["error"] == "NEED_PAPER_CONTEXT"
    assert result["primary_paper_id"] is None


def test_fetcher_uses_validated_request_and_returns_ids():
    arxiv = RecordingArxiv([paper("P003", "Three")])
    agent = FetcherAgent(arxiv, FakeParser(), FakeMongo(), FakeEmbedder(), FakeMilvus())
    result = agent.invoke(search_state(SearchRequest("arxiv_id", "P003")))
    assert arxiv.calls == [("P003", 1)]
    assert result["primary_paper_id"] == "P003"
    assert result["resolved_paper_ids"] == ["P003"]


def test_successful_result_updates_primary_focus():
    focus = PaperFocusState(primary_paper_id=None, active_paper_ids=[])
    updated = SessionStateReducer().reduce(focus, {
        "error": None,
        "primary_paper_id": "P003",
        "resolved_paper_ids": ["P003"],
    }, resolved_context("P003"))
    assert updated.primary_paper_id == "P003"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_direct_analyzer_selection.py tests/agents/test_fetcher_search_policy.py tests/core/test_session_state.py -q`

Expected: FAIL because DirectAnalyzer still searches arXiv and normalized fields/reducer are missing.

- [ ] **Step 3: Make DirectAnalyzer local-only**

Remove the `arxiv_api` constructor argument, `_fetch_and_analyze()`, `_download_pdf()`,
and every fallback call. At the start of `invoke`, read the primary ID from
`turn_context`; use legacy `target_paper_id` only as a compatibility fallback. If no ID
exists, return:

```python
{
    "answer": "请先选择或明确指定要分析的论文。",
    "error": "NEED_PAPER_CONTEXT",
    "primary_paper_id": None,
    "resolved_paper_ids": [],
}
```

Successful local analysis returns both `primary_paper_id` and
`resolved_paper_ids=[paper_id]`, plus legacy `resolved_paper_id` during migration.

- [ ] **Step 4: Guard Fetcher search and normalize results**

Read `turn_context.search_request`, deserialize it, and call only
`guarded_arxiv_search`. Use `max_results=1` for `arxiv_id`/`title` and 5 for keywords.
Return all fetched/already-existing IDs in `resolved_paper_ids`; set
`primary_paper_id` only when exactly one result was resolved. Convert admission and
validation errors to stable `EXTERNAL_SEARCH_NOT_ALLOWED` or `INVALID_SEARCH_REQUEST`
results.

- [ ] **Step 5: Remove Retriever identity guessing and propagate evidence IDs**

Delete `_extract_target_paper()`. Pass `turn_context.paper_ids` into the two-level
retrieval filters, and never choose the newest MongoDB paper. Retriever returns
`resolved_paper_ids` from retrieved chunks. Analyzer returns the same ordered unique
IDs and sets primary only when the turn context already has one or the evidence is from
exactly one paper. Presenter preserves these fields when returning an upstream answer.

- [ ] **Step 6: Implement the pure SessionStateReducer**

`SessionStateReducer.reduce(focus, agent_result, paper_context)` returns a new
`PaperFocusState`. It ignores failed/unresolved results, sets primary on a successful
specific result, keeps all comparison IDs, records source/confidence/time, and mirrors
the primary at the front of `active_paper_ids`. Add `normalize_agent_result()` to read
new fields first and legacy `resolved_paper_id` second.

- [ ] **Step 7: Run agent and reducer tests**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_direct_analyzer_selection.py tests/agents/test_fetcher_search_policy.py tests/core/test_session_state.py -q`

Expected: PASS.

- [ ] **Step 8: Commit agent safety changes**

```powershell
git add core/session_state.py agents/direct_analyzer.py agents/fetcher.py agents/retriever.py agents/analyzer.py agents/presenter.py tests/agents/test_direct_analyzer_selection.py tests/agents/test_fetcher_search_policy.py tests/core/test_session_state.py
git commit -m "fix: prevent implicit search during paper analysis"
```

### Task 5: Wire resolver and policy nodes into both workflows

**Files:**
- Modify: `core/deps.py`
- Modify: `graph/workflow.py`
- Modify: `main.py`
- Modify: `web/app.py`
- Create: `tests/graph/test_paper_context_workflow.py`

- [ ] **Step 1: Write a failing graph-order test**

```python
def initial_state(query: str, primary_id: str | None) -> dict:
    active_ids = [primary_id] if primary_id else []
    return {
        "user_query": query,
        "target_paper_id": None,
        "paper_focus": {
            "primary_paper_id": primary_id,
            "active_paper_ids": active_ids,
        },
        "recent_paper_contexts": [],
        "messages": [],
        "iteration": 0,
        "max_iterations": 1,
    }


class RecordingNode:
    def __init__(self, name: str, events: list[str], output: dict):
        self.name = name
        self.events = events
        self.output = output

    def invoke(self, _state):
        self.events.append(self.name)
        return dict(self.output)


def resolved_or_unresolved(primary_id: str | None) -> dict:
    return {
        "primary_paper_id": primary_id,
        "paper_ids": [primary_id] if primary_id else [],
        "status": "resolved" if primary_id else "unresolved",
        "source": "session_focus" if primary_id else "none",
        "confidence": 0.98 if primary_id else 0.0,
        "inherited": bool(primary_id),
        "switched_from_paper_id": None,
    }


def build_test_workflow(events: list[str], primary_id: str | None):
    context = resolved_or_unresolved(primary_id)
    resolver = RecordingNode("paper_context_resolver", events, {"paper_context": context})
    supervisor = RecordingNode("supervisor", events, {"intent": "analyze"})
    turn = RecordingNode(
        "turn_context",
        events,
        build_turn_context("总结实验结果", "analyze", context),
    )
    direct = RecordingNode("direct", events, {
        "answer": "analysis",
        "error": None,
        "primary_paper_id": primary_id,
        "resolved_paper_ids": [primary_id] if primary_id else [],
    })
    presenter = RecordingNode("presenter", events, {"answer": "done"})
    noop = RecordingNode("noop", events, {})
    return build_workflow(
        paper_context_resolver=resolver,
        supervisor=supervisor,
        turn_context_builder=turn,
        fetcher=noop,
        retriever=noop,
        analyzer=noop,
        critic=RecordingNode("critic", events, {"next_agent": "presenter"}),
        presenter=presenter,
        direct_analyzer=direct,
    )


def test_resolver_runs_before_supervisor_and_policy_before_agent():
    events = []
    workflow = build_test_workflow(events, primary_id="P001")
    workflow.invoke(initial_state("总结实验结果", "P001"))
    assert events[:4] == ["paper_context_resolver", "supervisor", "turn_context", "direct"]


def test_unresolved_analysis_never_enters_direct():
    events = []
    workflow = build_test_workflow(events, primary_id=None)
    result = workflow.invoke(initial_state("分析它的实验方法", None))
    assert "direct" not in events
    assert result["error"] == "NEED_PAPER_CONTEXT"
```

- [ ] **Step 2: Run graph tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/graph/test_paper_context_workflow.py -q`

Expected: FAIL because both workflows begin at Supervisor.

- [ ] **Step 3: Construct shared components**

`ServiceContainer.create_agents()` returns `paper_context_resolver`, `supervisor`, and
`turn_context` nodes before ordinary agents. Inject MongoDB and the existing LLM into
the resolver; inject MongoDB into the turn-context builder only for similar-paper title
fallback. Remove arXiv from DirectAnalyzer construction in `core/deps.py` and `main.py`.

- [ ] **Step 4: Change both graph structures**

Use:

```text
START -> paper_context_resolver -> supervisor -> turn_context
turn_context -> direct_analyzer | fetcher | retriever | presenter
```

Move conditional routing from Supervisor to `turn_context`. Keep downstream edges
unchanged. In the CLI, maintain an in-memory `PaperFocusState`, include it in every
initial state, and reduce the final result after each turn so the non-web entrypoint
obeys the same invariant.

- [ ] **Step 5: Run graph and state tests**

Run: `.venv\Scripts\python.exe -m pytest tests/graph/test_paper_context_workflow.py tests/state/test_graph_state.py -q`

Expected: PASS.

- [ ] **Step 6: Commit workflow integration**

```powershell
git add core/deps.py graph/workflow.py main.py web/app.py tests/graph/test_paper_context_workflow.py tests/state/test_graph_state.py
git commit -m "feat: resolve paper context across workflow entrypoints"
```

### Task 6: Persist focus and per-message paper metadata through one reducer

**Files:**
- Modify: `web/app.py`
- Modify: `tests/web/test_paper_selection.py`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_legacy_active_list_loads_as_primary_focus():
    session = session_from_document({
        "session_id": "s1",
        "title": "paper",
        "active_paper_ids": ["P001"],
        "messages": [],
    })
    assert session.paper_focus.primary_paper_id == "P001"


def test_initial_state_contains_recent_structured_paper_metadata():
    session = Session(id="s1", title="x")
    session.messages.append(ChatMessage(
        role="assistant",
        content="done",
        paper_context=resolved_context("P001"),
    ))
    state = create_web_initial_state("总结实验结果", session)
    assert state["recent_paper_contexts"] == [resolved_context("P001")]


def test_message_serialization_preserves_switch_metadata():
    message = ChatMessage(role="user", content="再看看 P002", paper_context=switch_context("P001", "P002"))
    payload = serialize_chat_message(message)
    assert payload["paper_context"]["switched_from_paper_id"] == "P001"
```

- [ ] **Step 2: Run web persistence tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/web/test_paper_selection.py -q`

Expected: FAIL because `Session.paper_focus` and `ChatMessage.paper_context` do not exist.

- [ ] **Step 3: Add backwards-compatible persistence**

Add `paper_context` to `ChatMessage` and `paper_focus` to `Session`. On load, prefer
stored `paper_focus`; otherwise derive primary from the first legacy active ID. On save,
persist `paper_focus.to_dict()` and mirror its active IDs to the legacy field. Include
message paper metadata in load, save, and API serialization. Build
`recent_paper_contexts` from the newest structured message records only.

- [ ] **Step 4: Replace ad hoc stream mutation with reducer application**

During `workflow.astream`, merge node updates into one accumulated result and retain
the latest normalized paper fields. After completion:

1. attach the resolver output to the current user message;
2. normalize the accumulated agent result;
3. call `SessionStateReducer.reduce()` exactly once;
4. mirror the reduced focus to compatibility fields;
5. attach the final paper context to the assistant message;
6. save the session.

Delete the current `if resolved_paper_id: session.active_paper_ids = [resolved_paper_id]` block.

- [ ] **Step 5: Run web persistence tests**

Run: `.venv\Scripts\python.exe -m pytest tests/web/test_paper_selection.py tests/web/test_chat_presentation.py -q`

Expected: PASS.

- [ ] **Step 6: Commit session reduction and metadata**

```powershell
git add web/app.py tests/web/test_paper_selection.py tests/web/test_chat_presentation.py
git commit -m "feat: persist resolved paper focus and message metadata"
```

### Task 7: Add the seven required regressions and verify the complete system

**Files:**
- Create: `tests/web/test_paper_context_regressions.py`
- Modify: any test helper files required by the fixtures, without changing production behavior.

- [ ] **Step 1: Implement the seven regression tests**

Use recording fake arXiv, MongoDB, and lightweight graph agents. Define a
`run_context_turn(query, initial_primary, explicit_target=None, search_results=None)`
test helper that executes `PaperContextResolver`, `SupervisorAgent`,
`TurnContextBuilder`, the guarded fake search when admitted, and
`SessionStateReducer`, returning the intermediate dictionaries, final focus, and
recorded arXiv calls. Assert all seven scenarios with one parameterized test:

```python
from types import SimpleNamespace


def run_context_turn(query, initial_primary, explicit_target=None, search_results=None):
    papers = [
        {"arxiv_id": "P001", "title": "Paper One", "abstract": ""},
        {"arxiv_id": "P002", "title": "Paper Two", "abstract": ""},
    ]
    repository = FakePaperRepository(papers)
    focus = PaperFocusState(
        primary_paper_id=initial_primary,
        active_paper_ids=[initial_primary] if initial_primary else [],
    )
    paper_context = PaperContextResolver(repository).resolve(
        query=query,
        explicit_target_paper_id=explicit_target,
        paper_focus=focus,
        recent_paper_contexts=[],
    )
    intent_output = SupervisorAgent(llm=None).invoke({
        "user_query": query,
        "paper_context": paper_context.to_dict(),
    })
    turn_output = TurnContextBuilder(repository).invoke({
        "user_query": query,
        "intent": intent_output["intent"],
        "paper_context": paper_context.to_dict(),
        "paper_focus": focus.to_dict(),
    })
    arxiv = RecordingArxiv(search_results or (
        [{"arxiv_id": "2401.00003", "title": "Paper Three"}]
        if "2401.00003" in query else []
    ))
    agent_result = {
        "error": turn_output.get("error"),
        "primary_paper_id": paper_context.primary_paper_id,
        "resolved_paper_ids": list(paper_context.paper_ids),
    }
    request_data = turn_output["turn_context"].get("search_request")
    if request_data:
        request = SearchRequest.from_dict(request_data)
        found = guarded_arxiv_search(
            arxiv,
            turn_output["turn_context"],
            request,
            1 if request.mode in {"arxiv_id", "title"} else 5,
        )
        found_ids = [item["arxiv_id"] for item in found]
        agent_result["resolved_paper_ids"] = found_ids
        agent_result["primary_paper_id"] = found_ids[0] if len(found_ids) == 1 else None
    final_focus = SessionStateReducer().reduce(
        focus,
        agent_result,
        paper_context,
    )
    return SimpleNamespace(
        intent=intent_output["intent"],
        turn_context=turn_output,
        final_focus=final_focus,
        arxiv_calls=arxiv.calls,
    )


@pytest.mark.parametrize(
    "query,initial_primary,explicit_target,expected_primary,expected_intent,search_calls",
    [
        ("请从方法、实验设置、消融和局限四方面结构化分析", "P001", None, "P001", "analyze", 0),
        ("总结实验结果", "P001", None, "P001", "analyze", 0),
        ("帮我找几篇类似论文", "P001", None, "P001", "search", 1),
        ("分析新论文", "P001", "P002", "P002", "analyze", 0),
        ("分析它的实验方法", None, None, None, "analyze", 0),
        ("搜索 arXiv:2401.00003 并分析", None, None, "2401.00003", "search", 1),
        ("帮我找 PINN Navier-Stokes 论文，并分析以下八个问题：方法、实验、局限", None, None, None, "search", 1),
    ],
)
def test_paper_context_regressions(
    query, initial_primary, explicit_target, expected_primary, expected_intent, search_calls
):
    result = run_context_turn(query, initial_primary, explicit_target)
    assert result.intent == expected_intent
    assert result.final_focus.primary_paper_id == expected_primary
    assert len(result.arxiv_calls) == search_calls
    if search_calls:
        assert result.arxiv_calls[0][0] != query


def test_analysis_without_focus_returns_business_error():
    result = run_context_turn("分析它的实验方法", None)
    assert result.turn_context["error"] == "NEED_PAPER_CONTEXT"


def test_long_search_uses_only_built_keywords():
    query = "帮我找 PINN Navier-Stokes 论文，并分析以下八个问题：方法、实验、局限"
    result = run_context_turn(query, None)
    assert result.arxiv_calls == [("PINN Navier-Stokes", 5)]
```

Each test checks paper context, normalized intent, `allow_external_search`, final focus,
exact arXiv call count, and exact arXiv query value where applicable.

- [ ] **Step 2: Run the required regression file**

Run: `.venv\Scripts\python.exe -m pytest tests/web/test_paper_context_regressions.py -q`

Expected: seven tests PASS.

- [ ] **Step 3: Run all paper-context-related tests**

Run: `.venv\Scripts\python.exe -m pytest tests/core/test_paper_context.py tests/core/test_turn_context.py tests/core/test_search_policy.py tests/core/test_session_state.py tests/agents/test_supervisor_session_focus.py tests/agents/test_direct_analyzer_selection.py tests/agents/test_fetcher_search_policy.py tests/graph/test_paper_context_workflow.py tests/web/test_paper_selection.py tests/web/test_chat_presentation.py tests/web/test_paper_context_regressions.py -q`

Expected: all selected tests PASS with zero failures.

- [ ] **Step 4: Run the complete suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS with zero failures.

- [ ] **Step 5: Run import, search-boundary, and diff checks**

Run: `.venv\Scripts\python.exe -c "from web.app import app; print(len(app.routes))"`

Expected: exit 0 and a positive route count.

Run: `rg -n "arxiv\.search\(" agents core graph web main.py`

Expected: no direct agent call except the single guarded implementation in
`core/search_policy.py`.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 6: Commit the regression suite**

```powershell
git add tests/web/test_paper_context_regressions.py
git commit -m "test: cover multi-turn paper context regressions"
```
