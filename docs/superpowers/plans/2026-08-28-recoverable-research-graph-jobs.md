# 可恢复研究图谱任务实施计划

> **供智能开发代理使用：** 按任务逐项执行；测试文件只保留在本地 `tests/`，不得加入 Git。

**目标：** 让所有已索引论文通过可终止、带租约、可回填的任务执行器生成当前版本研究图谱。

**架构：** MongoDB 负责任务租约、优先级、重试、版本和诊断；独立 Python 子进程执行单次 LLM 提取，父工作者负责截止时间与强制终止。启动对账存量论文，查询不触发同步生成。

**技术栈：** Python asyncio/subprocess、FastAPI、MongoDB/PyMongo、现有 LangChain LLM、原生 HTML/CSS/JavaScript、pytest。

---

### 任务 1：任务状态、租约与版本化回填

**文件：**

- 修改：`storage/research_graph.py`
- 本地测试：`tests/storage/test_research_graph_lifecycle.py`

- [ ] 写失败测试：旧版或无版本的 indexed 论文创建 backfill；当前版本完成任务不重复创建；无租约或租约过期的 extracting 被回收。
- [ ] 实现 `reconcile_indexed_papers`、`recover_expired_leases`、带优先级和租约的 `claim_next_job`。
- [ ] 实现 `completed_with_edges`、`completed_empty`、`retry_wait`、`failed` 及尝试历史。
- [ ] 实现手动重试与状态/任务列表查询。
- [ ] 运行 `pytest tests/storage/test_research_graph_lifecycle.py -q`，预期全部通过。

### 任务 2：可终止的独立提取进程

**文件：**

- 新建：`tools/research_graph_process.py`
- 修改：`agents/research_graph_extractor.py`
- 修改：`config.py`
- 本地测试：`tests/tools/test_research_graph_process.py`

- [ ] 写失败测试：超时子进程被终止；成功结果是严格 JSON；无效模型输出带诊断返回。
- [ ] 子进程从 stdin 读取论文/chunks，用现有 LLM 提取，通过 stdout 返回结果。
- [ ] 提取器返回候选与选中 chunk、解析和拒绝统计。
- [ ] 增加 `GRAPH_EXTRACTION_TIMEOUT_SECONDS`、租约、重试间隔和熔断配置。
- [ ] 运行 `pytest tests/tools/test_research_graph_process.py -q`，预期全部通过。

### 任务 3：调度、重试与熔断

**文件：**

- 修改：`tools/research_graph_worker.py`
- 修改：`tools/upload_worker.py`
- 修改：`web/app.py`
- 本地测试：`tests/tools/test_research_graph_worker.py`

- [ ] 写失败测试：PDF 队列忙时不领取；任务超时终止进程并进入 retry_wait；第二次失败进入 failed；队列继续下一篇。
- [ ] 父工作者启动子进程、续租、执行硬超时和强制终止。
- [ ] 基础设施连续失败触发熔断；成功清零。
- [ ] 启动时回收租约并对账所有 indexed 论文；上传完成创建新论文优先级任务。
- [ ] 运行 `pytest tests/tools/test_research_graph_worker.py -q`，预期全部通过。

### 任务 4：状态、诊断与手动重试界面

**文件：**

- 修改：`web/app.py`
- 修改：`web/static/graph.html`
- 本地测试：`tests/web/test_research_graph_api.py`

- [ ] 写失败测试：状态返回覆盖数与熔断；任务列表区分 completed_empty；失败任务可手动重试并唤醒工作者。
- [ ] 增加任务列表和手动重试 API。
- [ ] 页面展示触发规则、覆盖率、零关系原因、失败原因、租约回收与重试按钮。
- [ ] 运行 `pytest tests/web/test_research_graph_api.py -q`，预期全部通过。

### 任务 5：回归验证与提交

- [ ] 运行 `pytest -q`，预期全部本地测试通过。
- [ ] 运行 `python -m compileall web storage agents tools` 与 `git diff --check`，预期无错误。
- [ ] 确认 `git ls-files tests` 无输出，简历文件未暂存。
- [ ] 仅提交实现、中文设计和中文计划，提交信息使用中文。
