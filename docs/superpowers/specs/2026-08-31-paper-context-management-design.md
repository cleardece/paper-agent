# Paper Context Management Design

## Goal

Replace keyword-based follow-up detection with explicit paper context state so every
turn resolves its paper identity before routing intent, and no analysis request can
reach arXiv as an implicit fallback.

## Scope

This change implements the complete end-to-end flow:

```text
User Message
    -> PaperContextResolver
    -> PaperContext
    -> Supervisor / Intent Router
    -> TurnContext + Search Policy
    -> Agent
    -> AgentResult
    -> SessionStateReducer
    -> Session and message metadata
```

It covers single-paper analysis, inherited session focus, explicit paper switches,
multi-paper comparison, intentional external search, normalized agent results, and
backwards-compatible session persistence. It does not change paper parsing,
embedding, reranking, research-memory extraction, or knowledge-graph behavior except
where those callers must consume the new turn context.

## Root Cause

The current Supervisor inherits `active_paper_ids` only when the query contains one
of a short list of pronouns. A structured follow-up such as “总结实验结果” therefore
passes through LLM intent routing without a stable `target_paper_id`. DirectAnalyzer
then searches the local library by title, fails to find one, and forwards the raw
analysis request to arXiv. The active paper list also conflates the primary paper with
the set of papers involved in a comparison, and result-to-session updates are handled
ad hoc in the web stream.

## Architecture

### Paper context resolution

`PaperContextResolver` is the only component that identifies papers. It accepts the
raw query, an optional explicit frontend target, the session focus, recent message
paper metadata, and locally stored paper metadata. It returns:

```python
PaperContext(
    primary_paper_id: str | None,
    paper_ids: list[str],
    status: Literal["resolved", "unresolved", "ambiguous", "switch"],
    source: Literal[
        "explicit_target", "arxiv_id", "doi", "title_match",
        "session_focus", "history_resolution", "none",
    ],
    confidence: float,
    inherited: bool,
    switched_from_paper_id: str | None,
)
```

Resolution follows this deterministic order:

1. Explicit `target_paper_id` supplied by the frontend.
2. Local paper ID, arXiv ID, or DOI found in the current query.
3. Explicit title matches against the local paper library.
4. Explicit switch language paired with a resolvable paper candidate.
5. The session's unique `primary_paper_id`, unless the query explicitly requests a
   search or an unresolved switch.
6. Recent message `paper_context` metadata.
7. An unresolved or ambiguous result.

When multiple bounded local candidates remain ambiguous, an optional LLM resolver may
choose only from those candidate IDs. It cannot create a paper ID or authorize search.
The initial implementation exposes this seam but stays on the deterministic fast path
unless ambiguity actually exists.

If a session has a primary paper and the user does not explicitly search for or switch
to another paper, that primary paper is inherited regardless of pronouns. For “compare
the current paper with Paper B,” the current paper remains primary and both IDs appear
in `paper_ids`.

### Intent routing and turn context

Supervisor receives the resolved `PaperContext` and classifies only the action:
`analyze`, `rag`, `compare`, `search`, or `general`. It must not extract titles, choose
paper IDs, or mutate session focus.

The workflow then constructs:

```python
TurnContext(
    query: str,
    intent: Intent,
    primary_paper_id: str | None,
    paper_ids: list[str],
    paper_resolution_source: str,
    paper_resolution_confidence: float,
    allow_external_search: bool,
    search_request: SearchRequest | None,
)
```

Every downstream agent consumes `turn_context`. Existing scalar state fields remain
temporarily available only as compatibility projections populated from that object;
agents do not independently infer paper identity.

### Search admission and query construction

External search is a capability granted only to `search` intent. A
`SearchAdmissionGate` validates the turn before every arXiv call. A denied call raises
`ExternalSearchNotAllowed` and is converted to a stable business error rather than an
unhandled exception.

All accepted searches use:

```python
SearchRequest(
    mode: Literal["arxiv_id", "title", "keywords"],
    value: str,
)
```

`SearchQueryBuilder` constructs and validates this request from the raw message. It
extracts a canonical arXiv ID when present, an explicitly quoted or matched title when
present, or concise academic keywords for ordinary discovery requests. The raw user
message is never passed directly to arXiv. The builder rejects empty and excessively
long values.

Fetcher is the only workflow agent allowed to perform external discovery. Its arXiv
dependency is wrapped by the admission gate. DirectAnalyzer loses its arXiv dependency
and all download/search fallback behavior; it analyzes only resolved local papers. A
missing target returns `NEED_PAPER_CONTEXT`, and a stale explicit/local target returns
the existing not-found business error without changing focus.

### Agent results and session reduction

Agents return a normalized result projection:

```python
AgentResult(
    answer: str | None,
    error: str | None,
    primary_paper_id: str | None,
    resolved_paper_ids: list[str],
)
```

Existing fields such as retrieved chunks, evidence reports, and route state remain in
LangGraph state, but paper identity is always present in the normalized fields for a
successful paper-specific analysis.

`SessionStateReducer` is the only component that updates session paper focus. It runs
after the graph finishes and applies these rules:

- A successful paper-specific result sets `primary_paper_id` and includes it in the
  active set.
- A comparison preserves its primary paper and stores all resolved comparison IDs.
- A switch stores the new primary paper and switch metadata (`from` and `to`) on the
  current message context.
- An unresolved or failed result never silently overwrites a valid existing focus.
- A stale explicitly selected paper may be removed only when the repository confirms
  that exact ID no longer exists.

## Persistence Model

Sessions gain a nested focus object:

```python
PaperFocusState(
    primary_paper_id: str | None,
    active_paper_ids: list[str],
    source: str | None,
    confidence: float | None,
    last_resolved_at: datetime | None,
)
```

For backwards compatibility, a stored session without `paper_focus` is loaded by using
the first legacy `active_paper_ids` entry as primary. New saves persist `paper_focus`
and mirror `active_paper_ids` at the legacy top-level field until all existing readers
have migrated.

Each user and assistant message may persist:

```json
{
  "paper_context": {
    "primary_paper_id": "P001",
    "paper_ids": ["P001"],
    "status": "resolved",
    "source": "session_focus",
    "confidence": 0.98,
    "inherited": true,
    "switched_from_paper_id": null
  }
}
```

Old messages without this metadata load unchanged. Resolver history fallback reads
only structured paper metadata and does not mine assistant prose for guessed titles.

## Workflow Integration

Both graph construction paths (`graph/workflow.py` and the traced workflow in
`web/app.py`) add resolution before Supervisor. Initial state construction carries the
session focus and recent structured message metadata. Supervisor route output is
mapped from normalized intent to the existing agent graph:

- `analyze` -> DirectAnalyzer when a primary paper exists; otherwise Presenter with
  `NEED_PAPER_CONTEXT`.
- `compare` and `rag` -> Retriever.
- `search` -> Fetcher with an admitted and validated `SearchRequest`.
- `general` -> Presenter.

Fetcher may continue into Retriever after successful discovery. When the subsequent
analysis identifies one paper, its result establishes the primary focus; when a search
only returns multiple papers, the result preserves the existing primary focus and
records the returned IDs as resolved papers without choosing one arbitrarily.

Retriever filters by `turn_context.paper_ids` where paper-specific retrieval is
required. It removes its title-from-history heuristic and never substitutes the most
recent database paper for unresolved context.

## Invariants and Error Handling

The implementation enforces these invariants at executable boundaries:

1. `analyze` with no `primary_paper_id` cannot enter DirectAnalyzer and returns
   `NEED_PAPER_CONTEXT`.
2. A turn without external-search permission cannot call arXiv.
3. A successful analysis of a specific paper returns a non-null primary paper ID.
4. Every arXiv query is represented by a validated `SearchRequest` built by
   `SearchQueryBuilder`.
5. A valid session primary paper is inherited unless the current query explicitly
   searches for or switches to another paper.

Ambiguity returns `PAPER_CONTEXT_AMBIGUOUS` with candidate IDs retained internally.
Resolver/database failures return `PAPER_CONTEXT_RESOLUTION_FAILED` without admitting
search. Search builder validation failures return `INVALID_SEARCH_REQUEST`. Existing
parse, embedding, and storage failures retain their current behavior.

## Testing Strategy

Unit tests cover resolver priority, focus inheritance without pronouns, explicit
switches, multi-paper comparison, history metadata fallback, ambiguity, query building,
search denial, DirectAnalyzer's required-target invariant, and reducer behavior.

Workflow and web regression tests cover:

1. Session P001 plus a long non-pronominal analysis request resolves to P001,
   classifies as `analyze`, and makes zero arXiv calls.
2. “总结实验结果” inherits P001 and makes zero arXiv calls.
3. “帮我找几篇类似论文” classifies as `search` and admits external search.
4. An explicit P002 switches from P001 to P002.
5. No focus plus “分析它的实验方法” returns `NEED_PAPER_CONTEXT` and makes zero
   arXiv calls.
6. Searching and successfully analyzing P003 makes P003 the session primary paper.
7. A long mixed search-and-analysis message produces a concise validated search value,
   never the raw message.

Existing paper selection, retrieval, session persistence, and presentation tests remain
part of the verification suite. Tests use recording fake arXiv clients to assert exact
call counts and query values.

## Rollout and Compatibility

No database migration job is required. Session loading performs a lazy compatibility
projection, and the next save writes the new shape. The legacy fields remain readable
during this change. No new runtime dependency is introduced.
