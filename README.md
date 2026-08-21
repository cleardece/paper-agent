# Paper Agent

面向研究生论文阅读与小论文写作的研究助手。它将论文搜索、PDF 解析、混合检索和多 Agent 分析串成一条可追溯的研究工作流：当答案缺少本次检索到的论文证据时，系统会要求重检索或明确提示证据不足。

## 项目目标

- 搜索、导入并解析学术论文；
- 结合 MongoDB、Milvus 和 Hybrid Search 检索已入库论文；
- 用 LangGraph 编排路由、检索、分析、质量审核、呈现和反思；
- 为回答提供可核验的引用来源、质量状态和 Agent 执行时间；
- 作为长期个人研究工具，而非一次性的演示项目。

## 架构

```text
用户问题
  -> Supervisor
  -> Fetcher（搜索/入库）
     或 Retriever -> Analyzer -> Critic -> Presenter -> Reflector

PDF -> 解析 -> Chunk -> Embedding -> Milvus
论文与会话元数据 -> MongoDB
```

`Critic` 在 LLM 语义审核前先执行确定性规则：回答中的论文引用必须来自当前 `retrieved_chunks`。这项规则验证来源归属；它不替代对具体事实是否被原文蕴含的人工或 LLM 审核。

## 快速开始

### 前提

- Python 3.10 或更高版本；
- Docker Desktop；
- 一个兼容 OpenAI API 的 LLM endpoint；
- 若使用本地 Embedding，安装与本机 CUDA/CPU 兼容的 PyTorch。

### 配置与启动

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入 LLM_API_KEY，并按需修改模型和服务地址
docker compose --env-file .env up -d
python -m pip install -r requirements-dev.txt
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

打开 `http://localhost:8000`。也可以在 PowerShell 中运行：

```powershell
.\scripts\start.ps1 -Python python
```

`PA_DATA_ROOT` 控制 Docker 的本地数据目录，默认是仓库中的 `.local-data/`；它不会被 Git 跟踪。首次使用前请确保 Docker Desktop 有该目录的访问权限。

## 环境变量

| 变量 | 用途 |
| --- | --- |
| `LLM_MODEL`、`LLM_BASE_URL`、`LLM_API_KEY` | 对话、路由、分析和审核模型配置 |
| `MONGODB_URI`、`MONGODB_DB` | 论文、会话和记忆存储 |
| `MILVUS_HOST`、`MILVUS_PORT` | 向量检索服务 |
| `MINERU_URL`、`MINERU_BACKEND` | 可选的 MinerU 服务；无 GPU 时使用 `pipeline` |
| `USE_MCP`、`MCP_ARXIV_URL` | arXiv MCP 搜索优先级 |
| `SEMANTIC_SCHOLAR_API_KEY` | 可选的 Semantic Scholar 搜索 |
| `PA_DATA_ROOT` | Docker Compose 数据卷根目录 |

## 日常研究工作流

1. 在对话页搜索一个主题，或上传本地 PDF 入库。
2. 在论文库确认论文状态，点击“提问”回到会话中做单篇分析。
3. 对多篇已入库论文提问或比较，阅读回答下方的 Agent 时间线与证据面板。
4. 对用于写作的结论，打开论文原文核对上下文；系统提供的是来源约束和检索证据，不替代学术责任。
5. 将经人工核对的高价值问题加入本地评测集，持续观察检索和引用质量。

## 准确性边界

- 回答引用的论文标题必须属于当前检索结果；未知引用会触发重检索或“需要复核”状态。
- 没有检索到证据时，系统应明确拒答或说明证据不足，而不是依据模型常识补全。
- 规则校验只检查“引用来自哪里”，不证明每一条自然语言断言都被论文逐字蕴含。
- 用于小论文写作前，请自行打开原文核对实验设置、数据、数字和因果结论。

## 评测

复制示例并用你已入库论文的 arXiv ID 标注期望结果：

```powershell
Copy-Item evaluation/cases.example.json evaluation/cases.local.json
python scripts/evaluate.py --cases evaluation/cases.local.json
```

评测输出包括：

- `recall_at_5`：期望论文在前五个检索论文中的覆盖率；
- `citation_pass_rate`：回答引用通过来源规则校验的比例；
- `abstention_accuracy`：应拒答问题被正确拒答的比例；
- `latency_ms_p50`：端到端中位延迟。

`evaluation/cases.local.json` 和 `evaluation/results/` 被忽略，避免把个人研究主题和结果推送到公开仓库。

## 测试

```powershell
python -m pytest -q
docker compose --env-file .env.example config --quiet
python -B -c "import main; from web.app import app; print(len(app.routes))"
```

界面人工验收：切换深浅主题后刷新页面；查看会话和论文库之间主题是否保持；提交一次有证据的问答和一次无证据的问答；确认时间线显示节点耗时、证据面板状态明确、服务失败后发送按钮可再次使用。

## 数据与隐私

本地 PDF、向量库、Mongo 数据、缓存、上传文件和 API 密钥均不应提交。删除论文会影响本地 MongoDB/Milvus 数据，执行前请确认该论文不再需要。

## MinerU 资源策略

本项目默认只在需要解析 PDF 时启动本机 MinerU；单次解析结束后立即停止它，避免模型持续占用内存。下一次上传会自动重新启动 MinerU，因此会有一次模型加载等待，但不会改变解析质量。

若连续批量上传论文，可在 `.env` 中设置 `MINERU_IDLE_SHUTDOWN_SECONDS=300`，让 MinerU 空闲 5 分钟后再释放。该值默认是 `0`，优先保证普通设备在空闲时不被长期占用。

```dotenv
MINERU_IDLE_SHUTDOWN_SECONDS=0
MINERU_REQUIRE_ACCURATE_PARSE=true
MINERU_MEMORY_LIMIT=
MINERU_CPU_LIMIT=
```

`MINERU_MEMORY_LIMIT` 与 `MINERU_CPU_LIMIT` 都是可选的本机设置，例如 `MINERU_MEMORY_LIMIT=8g`、`MINERU_CPU_LIMIT=2.0`。留空表示不施加固定限制，避免开源项目假设所有用户有相同硬件。限制触发或 MinerU 出错时，默认严格模式会把论文标记为 `parse_failed`，不会静默用 pdfplumber 的普通解析结果入库；修复环境后可直接重新上传。

## 批量上传队列

上传按钮支持一次选择多份 PDF。任务会保存到 MongoDB，并严格串行执行；页面的“本批上传”面板会动态显示排队、解析、分块、索引、完成或失败状态，刷新页面后可恢复查看。

- 单篇上传完成解析后立即释放 MinerU；同一批次有两篇及以上时，MinerU 在该批次内保持热启动，最后一篇结束后释放。
- 每篇失败会自动重试一次；第二次失败会清理论文库中的元数据、chunks 和向量，仅保留原始 PDF 与失败任务记录，然后继续下一篇。
- 服务异常重启后，未完成任务会重新进入队列；已完成任务不重复解析。
- 默认限制可通过 `.env` 调整：`UPLOAD_BATCH_MAX_FILES=20`、`UPLOAD_MAX_FILE_MB=100`、`UPLOAD_QUEUE_MAX_PENDING=50`、`UPLOAD_JOB_RETENTION_DAYS=30`。队列不提供并发模式。

## 论文入库状态

- `chunked`：PDF 已解析，正文片段已保存到 MongoDB；可以直接进行单篇分析。系统会在需要时对已有 chunks 补索引，不会重新下载 PDF。
- `indexed`：chunk 向量和论文级向量都已写入 Milvus，可参与语义检索和跨论文问答。
- `embedding_failed` / `milvus_failed`：原始 chunks 被保留；后续从论文库分析时会尝试补索引。失败不会被伪装成 `indexed`。

论文库的“提问”会携带所选论文的稳定 ID，因此分析和补索引只会针对被点击的那篇论文，不按标题中的单个英文词猜测其他论文。

## 研究型多轮记忆

系统将记忆分为两类：

- **会话焦点**：当前论文、任务、待解决问题、最近对话与滚动摘要。连续提问“它的实验怎么复现”“第二个贡献是什么”会优先使用当前论文的稳定 ID；没有唯一焦点时会检索或请求澄清，不会从对话文本猜一个英文标题后重新搜索。
- **研究者档案**：学习情况、研究方向、研究项目和长期偏好。回答完成后，系统会从用户本人消息中静默提炼档案更新；每次变更保留来源会话、消息 ID、置信度、时间和前后版本。

可在 `GET /api/research-profile/local-user` 查看本地档案。对话中可直接说“研究方向改为……”“不是这样，……”来覆盖旧记录。研究者档案只用于理解用户意图，绝不会作为论文事实、检索 chunk 或引用来源。

## Git 约定

- 使用 Conventional Commit 前缀：`feat:`、`fix:`、`test:`、`docs:`、`chore:`；
- 一个提交只覆盖一个可说明、可验证的主题；
- 不重写已经推送的历史；
- IDE 配置、缓存、下载论文、测试输出和个人评测结果始终留在本地；
- 新功能提交前至少运行相关单元测试、导入 smoke test 和 `git diff --check`。
