# Library Paper Selection and Index Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Analyze the exact paper selected in the library, reuse its chunks, and resume its incomplete vector index.

**Architecture:** Pass `target_paper_id` from the library to the workflow. DirectAnalyzer resolves it before title matching, builds analysis text from MongoDB chunks, and atomically rebuilds incomplete Milvus vectors.

**Tech Stack:** FastAPI, LangGraph, MongoDB, Milvus, JavaScript, pytest.

---

### Task 1: Transport the stable library ID

**Files:**
- Modify: `web/static/papers.html`, `web/static/app.js`, `web/app.py`, `state/graph_state.py`
- Test: `tests/web/test_paper_selection.py`

- [ ] Write a failing test that constructs `ChatRequest(message="分析", target_paper_id="mmc")`, calls `create_web_initial_state("分析", target_paper_id="mmc")`, and asserts `state["target_paper_id"] == "mmc"`.
- [ ] Run `D:\\conda\\envs\\paper-agent\\python.exe -m pytest tests/web/test_paper_selection.py -v`; expect a constructor failure.
- [ ] Add optional `target_paper_id` to `ChatRequest` and `AgentState`; pass it from `/api/chat` into `create_web_initial_state`.
- [ ] Change `askAbout(title, arxivId)` to write `pendingPaperId`; change `sendMessage(text, targetPaperId)` to send `target_paper_id`; consume and clear both pending values together.
- [ ] Run the focused tests, then commit with `feat: carry selected paper id into chat`.

### Task 2: Resolve exact papers and chunks

**Files:**
- Modify: `agents/supervisor.py`, `agents/direct_analyzer.py`
- Test: `tests/agents/test_direct_analyzer_selection.py`

- [ ] Write a failing test using a fake `mmc` paper and one MongoDB chunk. Invoke with `target_paper_id="mmc"`; assert the analyzer receives `# Method\nmethod text` and the download spy remains zero.
- [ ] Add a failing unknown-ID test; assert error `selected_paper_not_found` and zero downloads.
- [ ] Run `D:\\conda\\envs\\paper-agent\\python.exe -m pytest tests/agents/test_direct_analyzer_selection.py -v`; expect failure.
- [ ] Resolve `state["target_paper_id"]` only with `mongo.get_paper`; do not fall back when absent. Add `_chunks_to_full_text()` that joins chunk content in `chunk_index` order under each `metadata.section` heading.
- [ ] Use stored `full_text` only as an optional cache; otherwise analyze chunks. Remove the Supervisor fallback that chooses the latest paper with no conversation context, and require normalized complete-title equality for non-ID library matching.
- [ ] Run the focused tests, then commit with `fix: analyze selected library chunks exactly`.

### Task 3: Recover incomplete indexes without duplication

**Files:**
- Modify: `storage/milvus.py`, `agents/direct_analyzer.py`
- Extend: `tests/agents/test_direct_analyzer_selection.py`

- [ ] Add failing tests where a `chunked` paper has zero chunk and paper vectors; assert it deletes stale vectors, writes fresh vectors, and becomes `indexed`. Add an embedding failure case asserting `embedding_failed` while Mongo chunks remain.
- [ ] Run the focused test file; expect failure.
- [ ] Add `count_paper_embeddings(arxiv_id)` to Milvus. Implement `_ensure_indexed(paper, chunks)`: compare vector counts, delete stale vectors only when recovery is needed, generate and insert vectors, then update `indexed`. Record `embedding_failed` or `milvus_failed` on the corresponding failure; never delete Mongo chunks.
- [ ] Run the focused tests, then commit with `feat: recover incomplete paper indexes`.

### Task 4: Document and verify contracts

**Files:**
- Modify: `README.md`, `tests/test_runtime_config.py`, `tests/web/test_static_ui.py`

- [ ] Add tests that the library carries the stable ID and the README explains `chunked`, `indexed`, and recovery.
- [ ] Document that `chunked` is analyzable Mongo content and `indexed` includes both Milvus vector levels.
- [ ] Run `D:\\conda\\envs\\paper-agent\\python.exe -m pytest -q` and `docker compose --env-file .env.example config --quiet`; expect all tests and Compose validation to pass.
- [ ] Commit with `docs: explain chunked index recovery`.
