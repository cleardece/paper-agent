# Contextual Query Rewriter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn context-dependent follow-ups into standalone retrieval queries while preserving paper scope and evidence boundaries.

**Architecture:** Add one best-effort agent between TurnContext and the existing route. Persist only its rewritten query in LangGraph state, then make Retriever and DirectAnalyzer consume that query without adding conversation text to evidence.

**Tech Stack:** Python, LangGraph, LangChain messages, pytest, FastAPI web state.

---

### Task 1: Define and test contextual rewriting

**Files:**
- Create: `core/contextual_query.py`
- Create: `tests/core/test_contextual_query.py`

- [ ] Test contextual rewrite, passthrough, and failure fallback.
- [ ] Implement a single-call JSON rewriter with bounded context.
- [ ] Run the focused test module.

### Task 2: Propagate state and consume the standalone query

**Files:**
- Modify: `state/graph_state.py`
- Modify: `web/app.py`
- Modify: `main.py`
- Modify: `agents/retriever.py`
- Modify: `agents/direct_analyzer.py`
- Modify: `tests/web/test_paper_selection.py`
- Modify: `tests/agents/test_direct_analyzer_selection.py`
- Create: `tests/agents/test_retriever_contextual_query.py`

- [ ] Add bounded prior dialogue and `retrieval_query` to state.
- [ ] Make both analysis paths consume `retrieval_query`.
- [ ] Verify history never enters retrieved evidence.

### Task 3: Wire both workflows and verify

**Files:**
- Modify: `graph/workflow.py`
- Modify: `core/deps.py`
- Modify: `web/app.py`
- Modify: `web/static/app.js`

- [ ] Insert the rewriter after TurnContext in CLI and web workflows.
- [ ] Expose the node's status without exposing raw context.
- [ ] Run only the focused context, workflow, and syntax checks.
