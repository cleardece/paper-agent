# Research Graph TPM Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make temporary graph LLM TPM exhaustion persistently recoverable so jobs retain progress and automatically finish after upstream capacity returns.

**Architecture:** Add a graph-only persistent admission schedule and rate-limit state to `ResearchGraphRepository`; route every graph LLM subprocess call through it in `ResearchGraphWorker`. Store extraction-phase checkpoints so a verification 429 resumes at verification, while permanent quota and content failures retain their current semantics.

**Tech Stack:** Python 3.12+, asyncio, MongoDB/PyMongo, LangChain `ChatOpenAI`, pytest.

---

### Task 1: Define graph-only TPM configuration

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Test: `tests/test_runtime_config.py`

- [ ] **Step 1: Add a focused failing configuration test**

Assert that graph configuration exposes an 8,000 TPM ceiling, 2,000 maximum output tokens, 1,000 prompt-overhead tokens, five-minute initial backoff, and one-hour maximum backoff.

- [ ] **Step 2: Run only the configuration test and verify it fails**

Run: `python -m pytest tests/test_runtime_config.py -q`
Expected: FAIL because the four graph settings do not exist.

- [ ] **Step 3: Add the configuration and bound graph completions**

Add `GRAPH_LLM_TPM_LIMIT`, `GRAPH_LLM_MAX_OUTPUT_TOKENS`, `GRAPH_LLM_PROMPT_OVERHEAD_TOKENS`, `GRAPH_RATE_LIMIT_BASE_DELAY_SECONDS`, and `GRAPH_RATE_LIMIT_MAX_DELAY_SECONDS`. Extend `_create_llm()` with an optional output-token limit and pass it only from `get_graph_llm()`.

- [ ] **Step 4: Run the focused configuration test**

Run: `python -m pytest tests/test_runtime_config.py -q`
Expected: PASS.

### Task 2: Implement durable rate-limit state and phase checkpoints

**Files:**
- Modify: `storage/research_graph.py`
- Test: `tests/storage/test_research_graph_lifecycle.py`

- [ ] **Step 1: Add failing repository lifecycle tests**

Cover these exact transitions: `defer_rate_limited()` refunds the claim attempt even at `max_attempts`; first 429 opens the scheduler immediately; admission reservations survive repository instances; legacy failed rate-limit jobs revive once; extraction checkpoints survive defer/resume and are cleared after batch completion.

- [ ] **Step 2: Run only the repository lifecycle tests and verify failure**

Run: `python -m pytest tests/storage/test_research_graph_lifecycle.py -q`
Expected: FAIL on missing repository methods.

- [ ] **Step 3: Add repository operations**

Implement `record_rate_limit()`, `defer_rate_limited()`, `revive_rate_limited_jobs()`, `reserve_llm_capacity()`, `effective_llm_tpm_limit()`, `record_llm_success()`, `save_extraction_checkpoint()`, and `get_extraction_checkpoint()`. Keep rate-limit history separate from business-attempt failure history and guard the legacy migration with `rate_limit_recovery_version=1`.

- [ ] **Step 4: Run the repository lifecycle tests**

Run: `python -m pytest tests/storage/test_research_graph_lifecycle.py -q`
Expected: PASS.

### Task 3: Route graph calls through admission control and resume checkpoints

**Files:**
- Modify: `tools/research_graph_worker.py`
- Test: `tests/tools/test_research_graph_worker.py`

- [ ] **Step 1: Replace the misleading TPM test with failing behavior tests**

Assert that `429001` calls `record_rate_limit()` and `defer_rate_limited()` but never `fail_attempt()`; waiting for a reserved slot renews the lease; a saved extraction checkpoint skips extraction and retries verification; oversized extraction and resolution payloads are split and their results merged.

- [ ] **Step 2: Run only the graph worker tests and verify failure**

Run: `python -m pytest tests/tools/test_research_graph_worker.py -q`
Expected: FAIL because the worker still uses generic retry handling.

- [ ] **Step 3: Add the graph call wrapper and checkpoint-aware phases**

Implement `_run_graph_call()` for token estimation, persistent reservation, heartbeat waiting, and success feedback. Route extraction, verification, Entity Resolution, and Fact Resolution through it. Persist candidates before verification and merge split results without changing external graph interfaces.

- [ ] **Step 4: Run only the graph worker tests**

Run: `python -m pytest tests/tools/test_research_graph_worker.py -q`
Expected: PASS.

### Task 4: Focused verification only

**Files:**
- Verify: `config.py`
- Verify: `storage/research_graph.py`
- Verify: `tools/research_graph_worker.py`

- [ ] **Step 1: Run the three directly affected test files together**

Run: `python -m pytest tests/test_runtime_config.py tests/storage/test_research_graph_lifecycle.py tests/tools/test_research_graph_worker.py -q`
Expected: PASS. Do not run the full regression suite, per user request.

- [ ] **Step 2: Compile only changed Python modules**

Run: `python -m py_compile config.py storage/research_graph.py tools/research_graph_worker.py`
Expected: exit code 0.

- [ ] **Step 3: Check patch integrity**

Run: `git diff --check`
Expected: no output and exit code 0.
