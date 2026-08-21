# 持久化批量论文上传队列设计

## 目标

允许用户一次选择多份 PDF，并以全局串行方式完成解析、分块、向量化和入库。任务进度必须在页面动态可见，服务重启后未完成工作可以恢复，且单篇失败不影响后续论文。

## 范围与非目标

- 支持一次选择多份 PDF，保留现有单一上传入口的兼容接口。
- 使用 MongoDB 持久化批次和单篇任务；不新增 Redis、Celery 或其他服务。
- 全局只允许一篇论文处于处理中，固定为并发 `1`。
- 上传后的原始 PDF 继续按当前项目策略保留，不新增“是否保留”选择项。
- 前端显示当前批次的可展开动态队列，使用轮询而非聊天 WebSocket。
- 系统按上传批次自动选择 MinerU 生命周期：单篇上传解析后立即释放；多篇批次在本批次连续解析间保持热启动，批次结束立即释放。
- 不在本期增加跨机器 worker、优先级队列、暂停/取消或手动重排。

## 数据模型

MongoDB 新增 `upload_batches` 和 `upload_jobs`。

`upload_batches`：

```json
{
  "batch_id": "uuid",
  "created_at": "datetime",
  "updated_at": "datetime",
  "total_count": 3
}
```

`upload_jobs` 每篇一条：

```json
{
  "job_id": "uuid",
  "batch_id": "uuid",
  "sequence": 0,
  "arxiv_id": "local_xxx",
  "filename": "paper.pdf",
  "pdf_path": "tmp_pdfs/<job_id>.pdf",
  "status": "queued | parsing | chunking | indexing | completed | failed | skipped",
  "stage_detail": "可读状态说明",
  "chunk_count": 0,
  "parse_source": "MinerU | pdfplumber | null",
  "attempt_count": 0,
  "max_attempts": 2,
  "error": null,
  "created_at": "datetime",
  "updated_at": "datetime",
  "finished_at": null
}
```

任务按 `created_at`、`sequence` 领取。`arxiv_id` 已存在于论文库或仍在队列时，任务直接记为 `skipped` 并保留原因，不进入解析。

## 队列与恢复

FastAPI 启动时创建一个单 Worker 协程：

1. 将上次异常退出遗留的 `parsing`、`chunking`、`indexing` 状态恢复成 `queued`，并写入恢复说明。
2. 原子领取最早的 `queued` 任务，更新为 `parsing`。
3. 复用既有 PDF 处理流程，并在解析、分块、索引阶段更新 job 状态。
4. 成功标记 `completed`，写入分块数和解析来源；可重试异常自动清理本次已写入的论文元数据、chunks 和 Milvus 向量后重试一次。
5. 第二次仍失败时，再次清理该论文元数据、chunks、chunk 向量和论文级向量，保留原始 PDF 与 `failed` job 的错误信息。
6. 继续领取下一条，任何时候只运行一篇。

上传 API 只负责校验、保存文件和创建任务，不直接启动每篇处理协程。进程内 `asyncio.Event` 用于唤醒 Worker；MongoDB 才是队列的真实来源，因此重启不会丢任务。

当一次提交只有一篇有效 PDF 时，Worker 在该篇解析结束后遵循 `MINERU_IDLE_SHUTDOWN_SECONDS=0` 的立即释放策略。当批次含两篇及以上有效 PDF 时，Worker 为该批次的连续解析保持 MinerU 热启动，并在该批次最后一篇解析结束后立刻释放；队尾后来加入的单篇不会改变当前批次策略。

## API 与限制

- `POST /api/uploads`：接收 `files: list[UploadFile]`，返回批次 ID、每个任务的初始状态和被拒绝项。
- `POST /api/upload`：保留为兼容入口，内部提交一个文件到相同队列。
- `GET /api/upload-batches/{batch_id}`：返回批次及按顺序排列的任务状态。

环境变量默认值：

```dotenv
UPLOAD_BATCH_MAX_FILES=20
UPLOAD_MAX_FILE_MB=100
UPLOAD_QUEUE_MAX_PENDING=50
UPLOAD_JOB_RETENTION_DAYS=30
```

上传前按文件数量、单文件大小和当前 `queued + processing` 数量检查。超出限制的文件不写盘、不创建任务，并在响应中给出原因。仅清理超过保留期的 `completed`、`failed`、`skipped` 任务；`queued` 和处理中任务始终保留。

## 前端交互

- 文件输入增加 `multiple`。
- 提交后在当前页出现一个默认展开的“本批上传”面板；用户可折叠它。
- 每秒请求批次 API，逐篇渲染文件名、状态、解析来源、分块数和失败原因。
- 页面刷新后，浏览器保存最近批次 ID 并恢复该面板；任务本身不依赖浏览器存在。
- 全部任务达到终态后停止轮询，但面板仍可查看结果。

## 错误处理与证据边界

- 非 PDF、超限文件、重复论文和队列已满均显示为单独的拒绝/跳过原因。
- 单篇解析、Embedding 或 Milvus 失败会自动重试一次。第二次失败后清除论文库中的半成品，仅保留原始 PDF、失败 job、阶段和错误原因；后续论文继续处理。
- 不修改论文问答、会话焦点或研究者记忆的证据边界。

## 验收标准

- 一次选中三份 PDF 后，三条任务按顺序显示；同一时刻最多一条为处理中。
- 第一篇失败后，第二篇仍继续；失败项显示阶段和错误原因。
- 可重试失败会自动执行两次；第二次失败后论文库中不存在该篇的元数据、chunks 和向量，但原始 PDF 与失败记录仍可定位。
- 重启 Web 服务后，原先处理中任务回到排队并继续，已完成任务不重复解析。
- 同名文件获得不同内部路径；已存在论文被标记跳过，不覆盖已有数据。
- 超过配置的文件数、大小或队列容量时，响应和面板显示拒绝原因。
- 刷新页面能恢复最近批次面板和最新进度。
- 批次结束后面板醒目汇总失败论文文件名、失败阶段和错误原因。
