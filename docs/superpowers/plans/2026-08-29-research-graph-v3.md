# 研究图谱 V3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 按本计划在当前任务内逐项执行；测试使用本地忽略文件，验证后删除临时测试代码。步骤使用复选框跟踪。

**目标：** 将研究图谱升级为全文分批抽取、自动质检、人工歧义队列和带双边证据的跨论文关联。

**架构：** `ResearchGraphExtractor` 负责确定性分批和两阶段 LLM 提示；独立子进程每次只执行一个提取或判定请求；Worker 串行编排批次并把检查点持久化到 MongoDB。Repository 原子替换系统边、保留人工状态并派生论文关联；FastAPI 和图谱页分开展示有效图谱与待确认项。

**技术栈：** Python、LangChain ChatOpenAI、asyncio subprocess、MongoDB/PyMongo、FastAPI、原生 HTML/CSS/JavaScript、pytest。

---

### 任务 1：全文分批与两阶段提取

**文件：**
- 修改：`agents/research_graph_extractor.py`
- 修改：`tools/research_graph_process.py`
- 临时测试：`tests/agents/test_research_graph_extractor.py`

- [ ] 写失败测试：超过字符预算的正文生成多个批次，参考文献被跳过，所有正文片段均被覆盖。
- [ ] 实现 `build_batches(chunks, batch_chars=12000, segment_chars=4000)`，长 chunk 拆段并保留 `chunk_index`、段号、标题和页码。
- [ ] 将单次子进程协议改为 `mode=extract|validate`。提取输出候选，判定输出 `supported|uncertain|rejected` 和原因。
- [ ] 提取提示要求 1～3 个连续完整原句；判定提示明确区分本文提出、本文使用、相关工作引用和单纯提及。
- [ ] 运行临时测试，确认批次覆盖和 JSON 解析通过。

关键接口：

```python
ResearchGraphExtractor.build_batches(chunks) -> list[list[dict]]
ResearchGraphExtractor.extract_batch(paper, batch) -> dict
ResearchGraphExtractor.validate_batch(paper, batch, candidates) -> dict
```

### 任务 2：V3 状态、严格证据和批次检查点

**文件：**
- 修改：`storage/research_graph.py`
- 临时测试：`tests/storage/test_research_graph_lifecycle.py`

- [ ] 将版本升级为 `evidence-graph-v3`，实体类型加入 `task`，关系类型加入 `improves`、`evaluates_on`、`measures_with`、`studies`。
- [ ] 新任务初始化 `batch_total`、`completed_batches`、`staged_relations` 和 `batch_diagnostics`。
- [ ] 实现批次检查点：

```python
save_batch_result(paper_id, worker_id, batch_index, batch_total,
                  relations, diagnostics) -> bool
```

- [ ] 将候选确定性分类为 `auto_verified`、`needs_review` 或自动拒绝；证据必须在原 chunk 中按空白归一化后精确定位。
- [ ] 完成全部批次后一次性替换系统生成边；`confirmed` 与 `rejected` 只更新证据字段，不覆盖人工状态。
- [ ] 图谱检索和 RAG 只使用 `auto_verified` 与 `confirmed`；人工队列只查询 `needs_review`。
- [ ] 运行生命周期临时测试，覆盖版本迁移、断点续跑、人工状态保护和系统边替换。

### 任务 3：Worker 分批编排与失败续跑

**文件：**
- 修改：`tools/research_graph_worker.py`
- 临时测试：`tests/tools/test_research_graph_worker.py`

- [ ] Worker 根据全文生成批次，跳过任务中已经完成的批次。
- [ ] 每批分别运行提取子进程和判定子进程，两个请求均受独立硬超时保护。
- [ ] 每批成功后立即保存检查点并续租；单批失败进入现有 `retry_wait`，下一轮从该批继续。
- [ ] 全部批次完成后读取暂存关系、原子写入边、清理暂存数据并完成任务。
- [ ] 诊断记录总批次数、已完成批次数、自动通过数、待确认数和自动拒绝数。
- [ ] 运行 Worker 临时测试，确认失败续跑不会重复已完成批次。

### 任务 4：跨论文关联和 RAG 使用边界

**文件：**
- 修改：`storage/research_graph.py`
- 修改：`agents/retriever.py`
- 修改：`web/app.py`
- 临时测试：`tests/storage/test_research_graph_lifecycle.py`

- [ ] 实现 `paper_links(limit=100)`：按共享 `target_node_id` 聚合两篇论文的有效边，返回实体、双方关系和双方证据。
- [ ] 增加 `GET /api/research-graph/paper-links`。
- [ ] 修改 `find_related_paper_ids`，排除 `needs_review` 与 `rejected`，避免未确认关系影响 RAG。
- [ ] 运行临时测试，确认共享实体产生双边证据且待确认边不参与检索。

### 任务 5：图谱页职责分离

**文件：**
- 修改：`web/static/graph.html`
- 临时测试：`tests/web/test_static_ui.py`

- [ ] 页面增加“论文关联”“有效关系”“待人工确认”三个区域。
- [ ] 默认有效关系只显示 `auto_verified` 和 `confirmed`；待确认区只显示 `needs_review`。
- [ ] 删除模型置信度百分比，改为“系统校验通过”“待人工确认”“人工已确认”。
- [ ] 证据区显示完整核心引文、章节、页码和可展开原 chunk 上下文。
- [ ] 待确认项保留确认/拒绝按钮；自动通过项允许抽查后确认或拒绝。
- [ ] 运行静态界面临时测试。

### 任务 6：清理、验证与提交

**文件：**
- 删除本轮新增的临时测试代码。
- 提交以上实现文件和本计划，不提交 `tests/` 与简历文件。

- [ ] 删除本轮新增测试，确认 `git ls-files tests` 无输出。
- [ ] 运行 `pytest -q`，预期原项目测试全部通过。
- [ ] 运行 `python -m compileall agents storage tools web`，预期退出码为 0。
- [ ] 运行 `git diff --check`，预期无错误。
- [ ] 使用中文提交信息创建本地提交，不推送远程仓库。
