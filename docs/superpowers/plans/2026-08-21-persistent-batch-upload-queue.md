# Persistent Batch Upload Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept multiple PDFs in one upload, process them through a restart-safe global serial queue, and render each batch's live result in the current page.

**Architecture:** Add an `UploadQueueRepository` over MongoDB for batches, jobs and atomic claims. A single FastAPI worker consumes one job at a time, checkpoints its phase, retries retryable failures once after cleanup, and uses an outer MinerU lease only for batches with multiple accepted jobs. The browser submits files to a batch endpoint and polls its persisted job state every second.

**Tech Stack:** FastAPI, asyncio, MongoDB/PyMongo, existing MinerU lifecycle manager, Milvus, vanilla HTML/CSS/JavaScript, pytest.

---

## File map

- Create: `storage/upload_queue.py` — MongoDB batch/job persistence and atomic queue operations.
- Modify: `config.py` and `.env.example` — portable batch/size/queue/retention configuration.
- Modify: `web/app.py` — batch endpoints, one worker, phase updates, retry cleanup and startup/shutdown lifecycle.
- Modify: `tools/mineru_lifecycle.py` — named outer batch lease that composes with existing parse leases.
- Modify: `web/static/index.html`, `web/static/app.js`, `web/static/style.css` — multi-select input and live expandable batch panel.
- Modify: `README.md` — queue limits, retention and failure semantics.
- Create: `tests/storage/test_upload_queue.py` — repository lifecycle and atomic-claim behavior using a fake collection.
- Create: `tests/web/test_batch_upload_queue.py` — API/job processing behavior with fake infrastructure.
- Modify: `tests/tools/test_mineru_lifecycle.py` — nested batch lease keeps the container alive until the outer lease exits.
- Modify: `tests/web/test_static_ui.py` — batch upload markup and browser polling contract.

### Task 1: Define portable queue limits and persistent job repository

**Files:**
- Modify: `config.py:132-154`
- Modify: `.env.example`
- Create: `storage/upload_queue.py`
- Create: `tests/storage/test_upload_queue.py`

- [ ] **Step 1: Write failing repository tests**

```python
def test_claim_next_job_uses_creation_then_sequence_order(fake_db):
    repo = UploadQueueRepository(fake_db)
    repo.create_batch("batch-1", [])
    repo.create_jobs("batch-1", [job("job-2", 1), job("job-1", 0)])

    claimed = repo.claim_next_job()

    assert claimed["job_id"] == "job-1"
    assert claimed["status"] == "parsing"


def test_recover_processing_jobs_requeues_without_losing_attempt_count(fake_db):
    repo = UploadQueueRepository(fake_db)
    repo.create_jobs("batch-1", [job("job-1", 0, status="indexing", attempt_count=1)])

    assert repo.requeue_interrupted_jobs() == 1
    assert repo.get_job("job-1")["status"] == "queued"
    assert repo.get_job("job-1")["attempt_count"] == 1
```

- [ ] **Step 2: Run the repository tests and confirm they fail**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/storage/test_upload_queue.py -q`

Expected: import error because `UploadQueueRepository` does not exist.

- [ ] **Step 3: Add configuration and implement the repository**

Add configuration constants with these defaults:

```python
UPLOAD_BATCH_MAX_FILES = int(os.getenv("UPLOAD_BATCH_MAX_FILES", "20"))
UPLOAD_MAX_FILE_MB = int(os.getenv("UPLOAD_MAX_FILE_MB", "100"))
UPLOAD_QUEUE_MAX_PENDING = int(os.getenv("UPLOAD_QUEUE_MAX_PENDING", "50"))
UPLOAD_JOB_RETENTION_DAYS = int(os.getenv("UPLOAD_JOB_RETENTION_DAYS", "30"))
```

Implement `UploadQueueRepository(db)` with `batches` and `jobs` collections. Create indexes on `batch_id`, `(status, created_at, sequence)`, `arxiv_id`, and `finished_at`. Define these methods:

```python
create_batch(batch_id: str, total_count: int) -> dict
create_jobs(batch_id: str, jobs: list[dict]) -> None
count_pending() -> int
claim_next_job() -> dict | None
update_job(job_id: str, status: str, *, detail: str = "", **fields) -> None
get_batch(batch_id: str) -> dict | None
get_job(job_id: str) -> dict | None
has_nonterminal_arxiv_id(arxiv_id: str) -> bool
requeue_interrupted_jobs() -> int
cleanup_terminal_jobs(retention_days: int) -> int
```

`claim_next_job` must call `find_one_and_update` with `{"status": "queued"}`, sorted by `created_at, sequence`, and atomically set status `parsing`, `attempt_count += 1`, detail `正在解析` and `updated_at`. Terminal statuses are `completed`, `failed`, and `skipped`.

Add matching commented values to `.env.example`.

- [ ] **Step 4: Run repository tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/storage/test_upload_queue.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the queue storage unit**

```bash
git add config.py .env.example storage/upload_queue.py tests/storage/test_upload_queue.py
git commit -m "feat: persist batch upload jobs"
```

### Task 2: Extend MinerU lifecycle with a batch-scoped outer lease

**Files:**
- Modify: `tools/mineru_lifecycle.py:45-82`
- Modify: `tests/tools/test_mineru_lifecycle.py`

- [ ] **Step 1: Write the failing nested-lease test**

```python
def test_batch_lease_keeps_mineru_running_across_two_parse_leases():
    commands = []
    manager = MinerUContainerManager("http://localhost:8888", command_runner=commands.append, health_check=lambda: True)

    with manager.batch_lease():
        with manager.lease():
            pass
        with manager.lease():
            pass

    assert commands == [
        ["docker", "compose", "up", "-d", "mineru-api"],
        ["docker", "compose", "stop", "mineru-api"],
    ]
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/tools/test_mineru_lifecycle.py -q`

Expected: attribute error because `batch_lease` does not exist.

- [ ] **Step 3: Implement `batch_lease` as a composable outer lease**

Add:

```python
@contextmanager
def batch_lease(self) -> Iterator[None]:
    """Keep MinerU alive while a multi-file batch is being processed."""
    with self.lease():
        yield
```

Do not modify `PDFParser.parse`; its existing inner `lease()` raises the active counter while parsing and releases back to the outer counter. With `MINERU_IDLE_SHUTDOWN_SECONDS=0`, only the final outer release stops the container.

- [ ] **Step 4: Run lifecycle tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/tools/test_mineru_lifecycle.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the MinerU lease unit**

```bash
git add tools/mineru_lifecycle.py tests/tools/test_mineru_lifecycle.py
git commit -m "feat: keep MinerU warm for batch parsing"
```

### Task 3: Refactor one-paper processing into checkpointed queue work

**Files:**
- Modify: `web/app.py:604-790`
- Create: `tests/web/test_batch_upload_queue.py`
- Modify: `tests/web/test_upload_parse_failure.py`

- [ ] **Step 1: Write failing processor tests**

```python
def test_retryable_index_failure_retries_once_then_cleans_partial_paper(fake_container, repo):
    job = queued_job("job-1", "batch-1", "local-1")
    worker = UploadQueueWorker(fake_container, repo)

    asyncio.run(worker.process_job(job))

    assert fake_container.milvus.delete_calls == ["local-1", "local-1"]
    assert fake_container.mongodb.deleted_papers == ["local-1", "local-1"]
    assert repo.get_job("job-1")["status"] == "failed"
    assert repo.get_job("job-1")["attempt_count"] == 2


def test_completed_job_reports_chunk_count_and_parser_source(fake_container, repo):
    job = queued_job("job-1", "batch-1", "local-1")

    asyncio.run(UploadQueueWorker(fake_container, repo).process_job(job))

    finished = repo.get_job("job-1")
    assert finished["status"] == "completed"
    assert finished["chunk_count"] == 2
    assert finished["parse_source"] == "MinerU"
```

- [ ] **Step 2: Run the processor tests and confirm they fail**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_batch_upload_queue.py -q`

Expected: import error because `UploadQueueWorker` does not exist.

- [ ] **Step 3: Implement worker and cleanup primitives**

Move `_process_upload`'s body into `UploadQueueWorker.process_job(job)`. Add explicit phase writes:

```python
repo.update_job(job_id, "parsing", detail="正在使用 MinerU 解析")
result = await loop.run_in_executor(None, container.pdf_parser.parse, pdf_path)
repo.update_job(job_id, "chunking", detail="正在生成论文片段", parse_source=result["source"])
chunks = container.pdf_parser.chunk(result["sections"])
repo.update_job(job_id, "indexing", detail="正在生成向量并入库", chunk_count=len(chunks))
```

After every exception call `cleanup_partial_paper(arxiv_id)`, which calls `mongodb.delete_paper(arxiv_id)` and `milvus.delete_by_paper(arxiv_id)` independently and logs either cleanup error. Retry exactly once for all `MinerUParseError`, embedding and Milvus exceptions; set a `retrying` detail before retry. On the second failure, set `failed`, error, `finished_at`, then return without raising. On success, set `completed`, `finished_at`, `chunk_count` and `parse_source`.

Keep `_process_upload` as a backward-compatible thin adapter that creates an in-memory job shape and delegates to the worker only for existing direct callers; change its strict MinerU failure test to assert `parse_failed` is no longer written for queued uploads.

- [ ] **Step 4: Implement a single durable queue loop**

Create `UploadQueueWorker.run()`:

```python
while not self.stopped:
    job = self.repository.claim_next_job()
    if job is None:
        await self.wakeup.wait()
        self.wakeup.clear()
        continue
    if self.repository.is_multi_file_batch(job["batch_id"]):
        with self._mineru_batch_lease():
            await self.process_contiguous_batch(job)
    else:
        await self.process_job(job)
```

`process_contiguous_batch` processes only remaining queued jobs from that batch in `sequence` order under one outer `batch_lease`; after the final job exits, the manager releases MinerU. New single-file batches remain queued behind it and do not extend that lease. On app startup call `requeue_interrupted_jobs`, `cleanup_terminal_jobs`, then start exactly one `asyncio.create_task(worker.run())`; on shutdown signal `worker.stop()` and await/cancel the task safely.

- [ ] **Step 5: Run queue worker and existing upload tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_batch_upload_queue.py tests/web/test_upload_parse_failure.py tests/tools/test_pdf_parser_mineru.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the worker unit**

```bash
git add web/app.py tests/web/test_batch_upload_queue.py tests/web/test_upload_parse_failure.py
git commit -m "feat: process uploads through a serial durable queue"
```

### Task 4: Add batch submission and status APIs

**Files:**
- Modify: `web/app.py:604-705, 790-820`
- Modify: `tests/web/test_batch_upload_queue.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_multiple_upload_creates_ordered_batch_and_distinct_internal_paths(client, fake_worker):
    response = client.post("/api/uploads", files=[
        ("files", ("same.pdf", b"%PDF-a", "application/pdf")),
        ("files", ("same.pdf", b"%PDF-b", "application/pdf")),
    ])

    body = response.json()
    assert response.status_code == 202
    assert body["accepted_count"] == 2
    assert body["jobs"][0]["pdf_path"] != body["jobs"][1]["pdf_path"]


def test_batch_status_returns_jobs_in_sequence_order(client, fake_repository):
    response = client.get("/api/upload-batches/batch-1")
    assert response.status_code == 200
    assert [job["sequence"] for job in response.json()["jobs"]] == [0, 1]
```

- [ ] **Step 2: Run API tests and confirm they fail**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_batch_upload_queue.py -q`

Expected: `/api/uploads` and `/api/upload-batches/{batch_id}` return 404.

- [ ] **Step 3: Implement upload submission**

Add `POST /api/uploads(files: list[UploadFile])`. For each accepted PDF:

1. Reject a non-PDF or over-size file with a per-file reason before writing.
2. Derive the existing stable `arxiv_id` from filename/content.
3. If `mongodb.get_paper(arxiv_id)` is a non-`parse_failed` paper or `repository.has_nonterminal_arxiv_id(arxiv_id)`, create a `skipped` job with `论文已存在或已在队列中`.
4. Save content to `tmp_pdfs/{job_id}.pdf`, never the browser filename.
5. Create a `queued` job with the original filename and sequence.

Reject any files beyond `UPLOAD_BATCH_MAX_FILES` and reject acceptance that would exceed `UPLOAD_QUEUE_MAX_PENDING`; return HTTP 202 with `batch_id`, `accepted_count`, `jobs`, and `rejected`. Set the worker wakeup event after the batch transaction. Keep `POST /api/upload` and translate its one `file` into the same submission helper, preserving its original response shape.

Add `GET /api/upload-batches/{batch_id}`. Return 404 for unknown IDs and otherwise return the batch plus its jobs in sequence order, excluding internal stack traces and `pdf_path` from public responses.

- [ ] **Step 4: Run API tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_batch_upload_queue.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the API unit**

```bash
git add web/app.py tests/web/test_batch_upload_queue.py
git commit -m "feat: accept persistent PDF upload batches"
```

### Task 5: Render a live, recoverable batch panel

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/app.js`
- Modify: `web/static/style.css`
- Modify: `tests/web/test_static_ui.py`

- [ ] **Step 1: Write failing static UI contract tests**

```python
def test_upload_supports_multiple_files_and_batch_progress_panel():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    script = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
    assert 'id="fileInput"' in html and "multiple" in html
    assert 'id="uploadBatchPanel"' in html
    assert 'fetch(`/api/upload-batches/${batchId}`)' in script
    assert 'localStorage.setItem("paperAgentLastUploadBatch"' in script
```

- [ ] **Step 2: Run static UI tests and confirm they fail**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -q`

Expected: missing multi-file input and batch-panel assertions.

- [ ] **Step 3: Implement submit, polling and rendering**

Change `<input id="fileInput">` to include `multiple`. Add a closed-by-user-capable `<details id="uploadBatchPanel">` below existing upload status, with a summary, batch count line and `#uploadBatchJobs` list.

In `app.js`, replace `files[0]` with `Array.from(fileInput.files)`, append every file as `files` to `FormData`, then send `POST /api/uploads`. On 202, store `batch_id` under `paperAgentLastUploadBatch`, render the initial jobs and call:

```javascript
async function refreshUploadBatch(batchId) {
  const response = await fetch(`/api/upload-batches/${batchId}`);
  if (!response.ok) return stopUploadPolling();
  const batch = await response.json();
  renderUploadBatch(batch);
  if (batch.jobs.some((job) => ["queued", "parsing", "chunking", "indexing"].includes(job.status))) return;
  stopUploadPolling();
}
```

Start `setInterval` at 1000 ms only while a job is non-terminal. On startup, read `paperAgentLastUploadBatch` and resume polling/rendering. Render statuses with text labels, parser source/chunk count after completion, and an error line for `failed` or `skipped`. The batch summary lists every failed filename and stage detail when terminal. Add CSS tokens compatible with the existing dark/light variables.

- [ ] **Step 4: Run static UI tests**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the UI unit**

```bash
git add web/static/index.html web/static/app.js web/static/style.css tests/web/test_static_ui.py
git commit -m "feat: show live batch upload progress"
```

### Task 6: Document and verify the full batch workflow

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Document limits and recovery behavior**

Add a README section covering batch queue limits, status meanings, one retry then cleanup, original-PDF retention, service-restart recovery, and automatic MinerU behavior for single versus multi-file submissions. Include the four environment variables and explicitly state that there is no concurrent parse mode.

- [ ] **Step 2: Run full automated verification**

Run: `D:\conda\envs\paper-agent\python.exe -m pytest -q`

Expected: all tests pass.

Run: `D:\conda\envs\paper-agent\python.exe -c "from web.app import app; print('route count:', len(app.routes))"`

Expected: exit 0 and a positive route count.

Run: `docker compose --env-file .env.example config --quiet`

Expected: exit 0.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 3: Commit documentation and verification-aligned changes**

```bash
git add README.md .env.example
git commit -m "docs: explain durable batch upload queue"
```
