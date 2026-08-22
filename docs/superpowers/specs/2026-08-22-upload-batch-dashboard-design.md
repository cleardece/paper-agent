# Upload Draft and Seven-Day Batch Dashboard Design

## Goal

Let a researcher select PDFs across several picker actions, review one editable draft, submit it as one durable serial-processing batch, and determine exactly which papers succeeded, failed, or were skipped without inferring that result from the paper library.

## Decisions

- Use the selected approach: server-backed recent-batch history, not browser-only history and not a separate history page.
- Keep the current page and upload entry point. Add three compact expandable regions below it: **待上传清单**, **本批上传**, and **近 7 天批次**.
- Keep the existing global serial worker, retry-once policy, cleanup-on-final-failure behavior, and MinerU single-versus-multi-file lifecycle policy unchanged.
- Show batches created during the trailing seven 24-hour periods. The server is the source of truth; browser local storage is only an optional pointer to the currently active batch.
- Do not persist an unsubmitted browser draft. Browsers cannot safely restore local `File` objects after a refresh, so a refresh clears only the draft and does not affect any submitted batch.

## User experience

### 1. Build a draft before submission

Each file-picker action appends valid PDF `File` objects to the **待上传清单** instead of sending a request immediately. A duplicate selected file, identified by the same `name`, `size`, and `lastModified`, appears only once. Every row shows filename and formatted size and has a remove action.

The action button reads `开始上传（N 篇）` and is disabled when the draft is empty. It sends all retained draft files to the existing multi-file upload endpoint in one request. A successful request creates exactly one batch for all accepted files, even when files were selected in several picker actions.

If the server rejects an individual file because it is not a PDF, exceeds a limit, or cannot enter the queue, the client retains that draft row, marks it `未提交` with the returned reason, and removes only accepted rows from the draft. Users can remove or retry rejected rows deliberately.

### 2. Show the submitted batch as it works

The **本批上传** panel opens after submission and remains associated with the batch returned by the server. It shows a summary line with total, queued/processing, completed, failed, and skipped counts. Each job row shows its filename, current stage, parser source and chunk count once complete, or its safe failure/skip reason when terminal.

The browser polls the existing per-batch endpoint every second while this batch contains a non-terminal job. `queued`, `parsing`, `chunking`, and `indexing` are non-terminal. `completed`, `failed`, and `skipped` are terminal. When all jobs are terminal, polling stops and the displayed final result remains available through the seven-day history.

### 3. Surface recent outcomes independently from the paper library

The **近 7 天批次** panel lists at most the 20 newest server-side batches created during the trailing seven 24-hour periods, newest first. A collapsed batch row includes created time, total count, and the completed/failed/skipped/active summary. Expanding a row reveals its individual jobs and their safe status details.

On page load, the client requests this list. While any listed batch is active it refreshes the list every second; otherwise it does not poll. The currently active batch continues to use the detail endpoint for timely progress. Refreshing the page, clearing local storage, or returning later still shows the seven-day history because it comes from MongoDB.

The paper library remains a view of successfully indexed papers only. It is explicitly not the upload-result ledger, and skipped/failed items appear only in the batch dashboard.

## Backend contract

The existing endpoint remains the source of a full single-batch view:

```text
GET /api/upload-batches/{batch_id}
```

Add a list endpoint:

```text
GET /api/upload-batches?days=7&limit=20
```

`days` defaults to `7`, accepts only integers from `1` through `7`, and `limit` defaults to `20`, capped at `20`. The response contains batches with their jobs in sequence order, each job exposing only:

```json
{
  "job_id": "...",
  "batch_id": "...",
  "sequence": 0,
  "arxiv_id": "...",
  "filename": "...",
  "status": "completed",
  "stage_detail": "已完成",
  "chunk_count": 15,
  "parse_source": "mineru",
  "attempt_count": 1,
  "max_attempts": 2,
  "error": null,
  "created_at": "...",
  "updated_at": "...",
  "finished_at": "..."
}
```

Neither endpoint returns `pdf_path`, original file content, internal stack traces, database IDs, or configuration values. The repository gains a `list_recent_batches(days, limit)` query that filters batches by `created_at >= now - days`, sorts descending by `created_at`, and attaches jobs ordered ascending by `sequence`.

No queue semantics change: the worker still claims only one job globally, retries a failed job once, cleans partial MongoDB/Milvus data on each failure, and moves to the next job after a final failure.

## Frontend state and rendering

The browser owns only two ephemeral state values:

```javascript
draftFiles: Map<string, File>
currentBatchId: string | null
```

The map key is `${file.name}:${file.size}:${file.lastModified}`. `currentBatchId` may be mirrored in local storage to resume a currently open batch detail after navigation, but history never depends on it.

The renderer derives all displayed counts from the job statuses returned by the server. It must not infer success from the presence of an item in `/api/papers`, nor optimisticly mark an item complete immediately after `POST /api/uploads` returns 202.

## Failure and boundary behavior

- Files are selected across several picker invocations: all visible draft items are submitted together as one batch.
- The same file is selected twice before submission: only one draft row appears.
- A job is `skipped`: it counts as skipped, never as successful, and its reason remains visible.
- A job fails after the automatic retry: it counts as failed and shows the persisted safe error. The worker continues with the next job as before.
- A batch has no accepted files: do not create a meaningless empty batch; keep the rejected draft rows visible with their reasons.
- A batch older than seven days: it stays subject to the existing configured job-retention policy but no longer appears in this dashboard.
- An endpoint request fails transiently: retain the last rendered state, show a non-blocking refresh warning, and try again on the next scheduled interval. Do not erase task results locally.

## Observability

The already planned indexing logs remain part of the queue work: emit phase durations for chunk embedding, chunk-vector Milvus insertion, paper-level embedding/insertion, and total worker time. These logs make an `indexing` row explainable without changing the persisted status protocol.

## Acceptance criteria

1. Choosing PDFs in two or more picker actions, then pressing one submit button, creates one batch with all accepted jobs in draft order.
2. The current page shows exact total, completed, active, failed, and skipped counts for that batch and names every failed or skipped PDF with its reason.
3. Reloading the page shows batches created in the last seven days and their outcomes, even when local storage has been cleared.
4. A paper library row is not used to calculate or display batch success.
5. Per-batch active progress refreshes without page reload and stops after terminal completion.
6. The global worker remains serial; retry, cleanup, and MinerU lifecycle tests retain their current behavior.
7. The new list API never exposes internal PDF paths, raw content, stack traces, or secrets.
