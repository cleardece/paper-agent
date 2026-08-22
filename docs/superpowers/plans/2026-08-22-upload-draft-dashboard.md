# Upload Draft and Seven-Day Batch Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users collect PDFs across several selections into one durable batch and view exact live and seven-day outcomes without treating the paper library as an upload ledger.

**Architecture:** Preserve the MongoDB-backed serial worker and the single-batch detail endpoint. Add a bounded recent-batch repository/API query, replace immediate browser submission with an in-memory draft, and add success-path timing logs without changing queue status transitions.

**Tech Stack:** FastAPI, PyMongo, asyncio, vanilla JavaScript/CSS, pytest.

---

## File map

- Modify: `storage/upload_queue.py` — list ordered batches created in a rolling seven-day window.
- Modify: `tests/storage/test_upload_queue.py` — cover time filtering and job order with the in-memory fake database.
- Modify: `web/app.py` — add a redacted, capped history endpoint and reuse response redaction.
- Create: `tests/web/test_upload_batch_api.py` — check endpoint bounds and that internal fields never escape.
- Modify: `tools/upload_worker.py` — add monotonic phase-duration logs only.
- Create: `tests/tools/test_upload_worker_observability.py` — test the successful indexing log sequence.
- Modify: `web/static/index.html`, `web/static/app.js`, `web/static/style.css` — draft, active batch, and history panels.
- Modify: `tests/web/test_static_ui.py` — assert the browser contract.

### Task 1: Query recent persisted batches

**Files:**
- Modify: `storage/upload_queue.py:105-120`
- Modify: `tests/storage/test_upload_queue.py`

- [ ] **Step 1: Write a failing repository test**

```python
def test_list_recent_batches_filters_window_and_orders_jobs(monkeypatch):
    repo = UploadQueueRepository(FakeDatabase())
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(repo, "_now", lambda: now)
    repo.create_batch("old", 1)
    repo.batches.docs[0]["created_at"] = now - timedelta(days=7, seconds=1)
    repo.create_batch("recent", 2)
    repo.create_jobs("recent", [job("second", 1), job("first", 0)])

    batches = repo.list_recent_batches(days=7, limit=20)

    assert [batch["batch_id"] for batch in batches] == ["recent"]
    assert [item["job_id"] for item in batches[0]["jobs"]] == ["first", "second"]
```

- [ ] **Step 2: Run it to verify the missing method fails**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/storage/test_upload_queue.py -q`

Expected: FAIL because `list_recent_batches` does not exist.

- [ ] **Step 3: Implement the rolling-window query**

```python
def list_recent_batches(self, days: int, limit: int) -> list[dict[str, Any]]:
    cutoff = self._now() - timedelta(days=days)
    batches = list(self.batches.find({"created_at": {"$gte": cutoff}}).sort([
        ("created_at", DESCENDING),
    ]).limit(limit))
    for batch in batches:
        batch["jobs"] = list(self.jobs.find({"batch_id": batch["batch_id"]}).sort([
            ("sequence", ASCENDING),
        ]))
    return batches
```

Extend `FakeCursor` with `limit(count)` and its matcher with `$gte` so the test stays Mongo-independent.

- [ ] **Step 4: Run the repository tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/storage/test_upload_queue.py -q`

Expected: all tests pass.

### Task 2: Serve redacted recent-batch history

**Files:**
- Modify: `web/app.py:742-752`
- Create: `tests/web/test_upload_batch_api.py`

- [ ] **Step 1: Write a failing endpoint test**

```python
def test_recent_upload_batches_are_capped_and_redacted(monkeypatch):
    repo = FakeQueueRepository([{
        "batch_id": "batch-1", "total_count": 1,
        "jobs": [{"job_id": "job-1", "filename": "paper.pdf", "status": "completed",
                  "pdf_path": "tmp_pdfs/private.pdf", "_id": "private"}],
    }])
    monkeypatch.setattr(web_app, "get_upload_queue", lambda: repo)

    response = asyncio.run(web_app.list_recent_upload_batches(days=99, limit=99))

    assert repo.arguments == (7, 20)
    assert "pdf_path" not in response["batches"][0]["jobs"][0]
    assert "_id" not in response["batches"][0]["jobs"][0]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_upload_batch_api.py -q`

Expected: FAIL because `list_recent_upload_batches` does not exist.

- [ ] **Step 3: Add shared response redaction and the endpoint**

```python
@app.get("/api/upload-batches")
async def list_recent_upload_batches(days: int = 7, limit: int = 20):
    safe_days = min(max(int(days), 1), 7)
    safe_limit = min(max(int(limit), 1), 20)
    batches = get_upload_queue().list_recent_batches(safe_days, safe_limit)
    return {"days": safe_days, "batches": [_public_upload_batch(batch) for batch in batches]}
```

`_public_upload_batch` must copy input data, remove batch `_id`, and remove `_id` and `pdf_path` from every job. Reuse it in `get_upload_batch`.

- [ ] **Step 4: Run endpoint and existing upload tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_upload_batch_api.py tests/web/test_upload_parse_failure.py -q`

Expected: all tests pass.

### Task 3: Make successful indexing observable

**Files:**
- Modify: `tools/upload_worker.py:48-110`
- Create: `tests/tools/test_upload_worker_observability.py`

- [ ] **Step 1: Write a failing log-order test**

```python
def test_completed_job_logs_indexing_boundaries_in_order(caplog, worker, job):
    with caplog.at_level(logging.INFO, logger="paper-agent"):
        asyncio.run(worker.process_job(job))
    messages = [record.getMessage() for record in caplog.records]
    expected = ["开始生成 2 个向量", "向量生成完成，耗时", "开始写入 Milvus（2 条）",
                "Milvus 写入完成，耗时", "开始写入论文级向量",
                "论文级向量写入完成，耗时", "任务完成，总耗时"]
    positions = [next(i for i, message in enumerate(messages) if phrase in message) for phrase in expected]
    assert positions == sorted(positions)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/tools/test_upload_worker_observability.py -q`

Expected: FAIL because the worker has no success-path phase timing logs.

- [ ] **Step 3: Add phase timing logs without changing state transitions**

Import `time`. Surround chunk `embed_texts`, chunk `milvus.insert`, and title embedding plus `insert_paper_embedding` with the seven test phrases and `time.monotonic()` elapsed values. Emit total duration immediately before the existing `completed` update. Do not log text, vectors, local paths, raw stack traces, or credentials.

- [ ] **Step 4: Run worker regressions**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/tools/test_upload_worker_observability.py tests/web/test_upload_parse_failure.py -q`

Expected: all tests pass.

### Task 4: Replace immediate submission with a draft and history dashboard

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/app.js:330-400`
- Modify: `web/static/style.css:150-160`
- Modify: `tests/web/test_static_ui.py`

- [ ] **Step 1: Write a failing browser contract test**

```python
def test_upload_ui_has_draft_submit_and_recent_history_contract():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    script = (ROOT / "web/static/app.js").read_text(encoding="utf-8")

    assert 'id="uploadDraftPanel"' in html
    assert 'id="startUploadBtn"' in html
    assert 'id="recentUploadBatches"' in html
    assert "const draftFiles = new Map()" in script
    assert 'fetch("/api/upload-batches?days=7&limit=20")' in script
    assert 'fileInput.addEventListener("change", addFilesToDraft)' in script
```

- [ ] **Step 2: Run it to verify it fails**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -q`

Expected: FAIL because file selection currently posts immediately.

- [ ] **Step 3: Add the dashboard markup and styles**

Add hidden `#uploadDraftPanel` with `#uploadDraftSummary`, `#uploadDraftFiles`, and disabled `#startUploadBtn`. Add `#recentUploadBatches` after the existing active batch panel. Use existing theme tokens for draft rows, remove controls, status colors and a non-blocking refresh warning; preserve the current mobile single-column layout.

- [ ] **Step 4: Implement browser state and rendering**

Define `const draftFiles = new Map()` keyed by `${file.name}:${file.size}:${file.lastModified}`. `addFilesToDraft` appends unseen PDFs, clears the file input, and never fetches. `startUploadBtn` sends all `draftFiles.values()` as `files` to `POST /api/uploads`.

After HTTP 202, remove only filenames present in returned `jobs`, leave rejected draft rows marked with their returned reason, save `paperAgentLastUploadBatch`, and render the returned batch. Derive all counts from `job.status`. Add `refreshRecentUploadBatches()` for exactly `GET /api/upload-batches?days=7&limit=20`; render newest-first expandable batches. Poll both active-batch detail and history every second while either response has a non-terminal job, then stop one shared timer. On a failed refresh retain the last DOM and show a warning.

- [ ] **Step 5: Run static UI tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -q`

Expected: all tests pass.

### Task 5: Verify and commit

**Files:**
- Modify: `README.md` only if its queue section needs one sentence explaining the seven-day dashboard.

- [ ] **Step 1: Run full automated verification**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest -q`

Expected: all tests pass.

Run: `D:\conda\envs\paper-agent\python.exe -c "from web.app import app; print('route count:', len(app.routes))"`

Expected: exit 0 and a positive route count.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 2: Perform the user-owned browser acceptance check**

Select two PDFs, use the picker again to add another, remove one draft item, then submit the remaining two. Confirm a single batch shows exact terminal counts, refresh the page, and confirm that same batch appears under **近 7 天批次**. The user performs this test; do not claim it passed until they report the result.

- [ ] **Step 3: Commit the implementation**

```bash
git add storage/upload_queue.py web/app.py tools/upload_worker.py web/static/index.html web/static/app.js web/static/style.css tests/storage/test_upload_queue.py tests/web/test_upload_batch_api.py tests/web/test_static_ui.py tests/tools/test_upload_worker_observability.py README.md && git commit -m "feat: add upload batch dashboard"
```
