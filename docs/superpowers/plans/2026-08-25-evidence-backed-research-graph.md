# 证据约束研究图谱实施计划

> **供智能开发代理使用：** 按任务逐项实现与验证；每个任务均采用测试先行。

**目标：** 为已索引论文建立持久、可追溯到原文片段的研究关系图谱，并让它辅助跨论文检索，而不影响原有 PDF 入库和 RAG。

**架构：** MongoDB 保存节点、边和可恢复任务；论文向量入库后仅投递图谱任务。独立单工作者只在 PDF 队列空闲时使用现有 LLM 提取高信号片段中的关系；图谱只缩小 RAG 的候选论文，最终回答仍以原文 chunk 为唯一事实依据。

**技术栈：** Python、FastAPI、MongoDB/PyMongo、LangChain OpenAI 兼容 LLM、Milvus/BGE-M3、pytest、原生 HTML/CSS/JavaScript。

---

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `storage/research_graph.py` | 节点、关系边、任务的 MongoDB 持久化、校验和查询 |
| `agents/research_graph_extractor.py` | 高信号 chunk 选择、严格 JSON 提取和候选校验 |
| `tools/research_graph_worker.py` | PDF 队列空闲时串行执行图谱任务、重试一次 |
| `core/deps.py` | 注入图谱仓储给上传、检索和 Web 层 |
| `tools/upload_worker.py` | 论文 indexed 后投递图谱任务；清理半成品图谱 |
| `web/app.py` | 启停工作者、图谱 API、删除论文时清理自动关系、图谱页路由 |
| `agents/retriever.py` | 将图谱命中的论文 ID 作为现有两层检索的可选过滤条件 |
| `web/static/graph.html` | 中文证据优先关系浏览页 |
| `tests/storage/test_research_graph.py` | 仓储、审核、清理、证据约束 |
| `tests/agents/test_research_graph_extractor.py` | 片段筛选、JSON 解析与候选过滤 |
| `tests/tools/test_research_graph_worker.py` | 空闲调度、一次重试、失败隔离 |
| `tests/web/test_research_graph_api.py` | 查询、状态和审核接口 |

### 任务 1：建立可验证的图谱仓储

**文件：**

- 新建：`tests/storage/test_research_graph.py`
- 新建：`storage/research_graph.py`

- [ ] **步骤 1：写失败测试，定义每条关系必须绑定本论文的有效原文证据。**

```python
def test_upsert_relations_rejects_unknown_or_too_short_evidence(graph_repo):
    saved = graph_repo.upsert_relations(
        paper={"arxiv_id": "p1", "title": "Paper One"},
        chunks=[{"chunk_index": 2, "content": "A validated method uses Dataset X."}],
        relations=[
            {"relation": "uses", "target_type": "dataset", "target_name": "Dataset X",
             "evidence_chunk_index": 9, "evidence": "not in this paper", "confidence": 0.9},
            {"relation": "uses", "target_type": "dataset", "target_name": "Dataset X",
             "evidence_chunk_index": 2, "evidence": "too short", "confidence": 0.9},
        ],
    )
    assert saved == []
```

- [ ] **步骤 2：运行 `pytest tests/storage/test_research_graph.py -q`，确认模块缺失而失败。**
- [ ] **步骤 3：实现 `ResearchGraphRepository`。** 创建 `research_graph_nodes`、`research_graph_edges`、`research_graph_jobs`；边记录 `source_paper_id`、chunk/section/page、内容 hash、置信度、提取版本与 `auto/confirmed/rejected` 审核状态。只接受 `proposes/uses/compares_with` 与 `method/dataset/metric`，稳定 ID 由论文、关系、目标和证据 chunk 计算。
- [ ] **步骤 4：补充以下审核保留测试并运行同一命令。**

```python
def test_confirmed_edge_survives_automatic_paper_cleanup(graph_repo):
    edge = graph_repo.upsert_relations(PAPER, CHUNKS, [VALID_RELATION])[0]
    assert graph_repo.review_edge(edge["_id"], "confirmed") is True
    graph_repo.delete_auto_data_for_paper("p1")
    assert graph_repo.get_edge(edge["_id"])["review_status"] == "confirmed"
```

预期：全部通过。

- [ ] **步骤 5：提交。**

```bash
git add storage/research_graph.py tests/storage/test_research_graph.py
git commit -m "feat: add evidence-backed graph repository"
```

### 任务 2：提取并验证带原文依据的关系

**文件：**

- 新建：`tests/agents/test_research_graph_extractor.py`
- 新建：`agents/research_graph_extractor.py`

- [ ] **步骤 1：写失败测试，限定高信号输入与 JSON 错误处理。**

```python
def test_extractor_selects_method_result_chunks_and_rejects_invalid_json(fake_llm):
    extractor = ResearchGraphExtractor(fake_llm, max_chunks=2)
    candidates = extractor.extract(PAPER, [INTRO, METHOD, RESULTS, REFERENCES])
    assert fake_llm.last_prompt.count('"chunk_index"') <= 2
    assert candidates == []
```

- [ ] **步骤 2：运行 `pytest tests/agents/test_research_graph_extractor.py -q`，确认导入失败。**
- [ ] **步骤 3：实现 `ResearchGraphExtractor`。** 选择最多 12 个标题匹配 abstract、introduction、method、experiment、result、evaluation、conclusion、limitation 的 chunk；提示模型只返回 JSON 数组，字段固定为关系、目标类型/名称、证据 chunk、证据文本与置信度。使用现有模型、`temperature=0`；解析代码围栏 JSON，并拒绝未知类型、空实体、越界置信度、非本论文 chunk、与 chunk 无非平凡文本重叠的候选。
- [ ] **步骤 4：为成功 JSON、代码围栏、无效证据添加测试并运行 `pytest tests/agents/test_research_graph_extractor.py -q`。**

预期：全部通过。

- [ ] **步骤 5：提交。**

```bash
git add agents/research_graph_extractor.py tests/agents/test_research_graph_extractor.py
git commit -m "feat: extract graph relations with evidence validation"
```

### 任务 3：实现低优先级、可恢复的图谱工作者

**文件：**

- 新建：`tests/tools/test_research_graph_worker.py`
- 新建：`tools/research_graph_worker.py`
- 修改：`core/deps.py`、`tools/upload_worker.py`、`web/app.py`

- [ ] **步骤 1：写失败测试，要求 PDF 队列有工作时不领取图谱任务。**

```python
async def test_worker_does_not_claim_graph_job_while_pdf_queue_is_busy(worker, upload_queue):
    upload_queue.count_pending.return_value = 1
    await worker.process_once()
    worker.graph_repository.claim_next_job.assert_not_called()
```

- [ ] **步骤 2：运行 `pytest tests/tools/test_research_graph_worker.py -q`，确认导入失败。**
- [ ] **步骤 3：实现单工作者。** `process_once()` 先检查 `upload_queue.count_pending()`；无 PDF 工作才 claim 图谱任务，读取 indexed 论文和 chunk，调用提取器并写边。失败第一次重新排队，第二次标记 `failed` 与错误文本；无论失败与否都不修改 indexed 论文、chunk 或 Milvus 向量。上传 indexed 后 `enqueue(arxiv_id)` 并唤醒图谱工作者；启动时为既有 indexed 论文补建缺失任务；删除论文和上传失败清理自动边，保留 confirmed 边。
- [ ] **步骤 4：加入一次重试隔离测试并运行测试。**

```python
async def test_worker_retries_once_then_marks_failed_without_touching_indexed_paper(worker):
    worker.extractor.extract.side_effect = RuntimeError("LLM timeout")
    assert await worker.process_once() is True
    assert await worker.process_once() is True
    assert worker.graph_repository.last_job_status == "failed"
    assert worker.mongodb.get_paper("p1")["status"] == "indexed"
```

运行：`pytest tests/tools/test_research_graph_worker.py -q`

预期：全部通过。

- [ ] **步骤 5：提交。**

```bash
git add core/deps.py tools/upload_worker.py tools/research_graph_worker.py web/app.py tests/tools/test_research_graph_worker.py
git commit -m "feat: process graph jobs after indexing when idle"
```

### 任务 4：图谱辅助跨论文 RAG

**文件：**

- 新建：`tests/agents/test_retriever_graph_filter.py`
- 修改：`agents/retriever.py`、`core/deps.py`

- [ ] **步骤 1：写失败测试，确保图谱只过滤候选论文、不产生回答。**

```python
def test_retriever_uses_graph_paper_ids_as_optional_filter(retriever, graph_repository):
    graph_repository.find_related_paper_ids.return_value = ["p1", "p3"]
    retriever.invoke({"user_query": "哪些论文使用 Dataset X？", "session_id": "s1"})
    assert retriever.milvus.search_chunks.call_args.kwargs["paper_ids"] == ["p1", "p3"]
    assert retriever.last_result[0]["content"] == "原论文片段"
```

- [ ] **步骤 2：运行 `pytest tests/agents/test_retriever_graph_filter.py -q`，确认过滤参数尚不存在。**
- [ ] **步骤 3：扩展 `_two_level_retrieval(..., graph_paper_ids=None)`。** 查询含“哪些论文、共同使用、比较、关系”等跨论文意图且图谱命中时才添加过滤；普通检索、单篇 direct 和无图谱命中保持原行为。状态中增加 `graph_context`，用于说明已用图谱定位候选论文，但 Analyzer 和 Presenter 仍只根据 chunk 陈述事实。
- [ ] **步骤 4：运行 `pytest tests/agents/test_retriever_graph_filter.py tests/agents/test_retriever.py -q`。**

预期：全部通过。

- [ ] **步骤 5：提交。**

```bash
git add agents/retriever.py core/deps.py tests/agents/test_retriever_graph_filter.py
git commit -m "feat: use graph candidates to assist retrieval"
```

### 任务 5：图谱 API 与中文证据浏览页

**文件：**

- 新建：`tests/web/test_research_graph_api.py`
- 新建：`web/static/graph.html`
- 修改：`web/app.py`、`web/static/papers.html`、`web/static/style.css`

- [ ] **步骤 1：写失败 API 测试。**

```python
def test_graph_search_returns_edges_with_evidence(client, graph_repository):
    response = client.get("/api/research-graph?query=Dataset%20X")
    assert response.status_code == 200
    edge = response.json()["edges"][0]
    assert edge["evidence"]["content"] == "A validated method uses Dataset X."

def test_review_edge_accepts_only_explicit_review_states(client):
    assert client.patch("/api/research-graph/edges/e1", json={"review_status": "confirmed"}).status_code == 200
    assert client.patch("/api/research-graph/edges/e1", json={"review_status": "maybe"}).status_code == 422
```

- [ ] **步骤 2：运行 `pytest tests/web/test_research_graph_api.py -q`，确认路由返回 404。**
- [ ] **步骤 3：实现 `GET /graph`、`GET /api/research-graph`、`GET /api/research-graph/status` 与 `PATCH /api/research-graph/edges/{edge_id}`。** 页面默认用关系卡而不是无证据的蛛网图：可按实体、关系和审核状态筛选；每张卡展示“论文 → 关系 → 实体”、置信度、状态、可展开原文证据；确认/拒绝即时更新。论文库增加图谱入口，并显示待提取、提取中、已就绪、失败状态。
- [ ] **步骤 4：运行 `pytest tests/web/test_research_graph_api.py -q; python -m compileall web storage agents tools`。**

预期：测试通过，编译无错误。

- [ ] **步骤 5：提交。**

```bash
git add web/app.py web/static/graph.html web/static/papers.html web/static/style.css tests/web/test_research_graph_api.py
git commit -m "feat: add evidence-first research graph explorer"
```

### 任务 6：端到端回归与交付检查

**文件：**

- 修改：`docs/superpowers/plans/2026-08-25-证据约束研究图谱.md`

- [ ] **步骤 1：运行完整回归。**

运行：`pytest -q`

预期：全部通过。

- [ ] **步骤 2：执行静态验证。**

运行：`python -m compileall web storage agents tools; git diff --check; Select-String -Path docs/superpowers/plans/2026-08-25-evidence-backed-research-graph.md -Pattern 'TODO|TBD|implement later|fill in details'`

预期：编译和 diff 检查无输出，关键词搜索无匹配。

- [ ] **步骤 3：提交最终计划勾选状态与实现。**

```bash
git add docs/superpowers/plans/2026-08-25-证据约束研究图谱.md
git commit -m "docs: record research graph implementation plan"
```
