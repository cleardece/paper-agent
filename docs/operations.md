# 运行与配置参考

[返回项目首页](../README.md) · [技术架构](architecture.md)

本文保留部署、解析、队列与评测的操作细节。命令均在仓库根目录执行。

## 快速开始（Quick Start）

### 前提

- Python 3.10 或更高版本；
- Docker Desktop；
- 一个兼容 OpenAI API 的 LLM endpoint；
- MinerU 官方 API Token；示例配置中的图谱另需本机 Ollama 和 `qwen3-graph:8b` 模型；
- 若使用本地 Embedding，安装与本机 CUDA/CPU 兼容的 PyTorch。

### 配置与启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
# 编辑 .env，填入 LLM_API_KEY、MINERU_OFFICIAL_TOKEN，并核对服务地址
python -m pip install -r requirements.txt
ollama pull qwen3:8b
ollama create qwen3-graph:8b -f Modelfile
docker compose --env-file .env up -d
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

以上命令在仓库根目录执行，Ollama 服务需保持运行。开发时可安装 `requirements-dev.txt` 并按需使用 `--reload`。默认配置面向本机使用。

打开 `http://localhost:8000`。也可以在 PowerShell 中运行：

```powershell
.\scripts\start.ps1 -Python python
```

`PA_DATA_ROOT` 控制 Milvus 及其依赖的数据目录，默认是仓库中的 `.local-data/`；MongoDB 使用 Docker 命名卷。`.local-data/` 不会被 Git 跟踪。首次使用前请确保 Docker Desktop 有该目录的访问权限。

## 环境变量说明

复制 [.env.example](../.env.example) 后编辑本机 `.env`。“需配置”表示按实际服务填写，已有默认值的项目可沿用示例。

| 变量 | 是否必填 | 用途 / 示例配置 |
| --- | --- | --- |
| `LLM_MODEL`、`LLM_BASE_URL` | 是，需核对 | 对话、路由、分析和审核的模型与 OpenAI 兼容地址 |
| `LLM_API_KEY` | 是，需配置 | LLM 凭据；本地无鉴权服务按其接口要求填写 |
| `MINERU_OFFICIAL_TOKEN` | **是，需配置** | MinerU 官方 Token；缺失时应用拒绝启动 |
| `MINERU_OFFICIAL_BASE_URL` | 否，已有默认值 | `https://mineru.net`；解析固定使用 `vlm` |
| `MINERU_OFFICIAL_POLL_SECONDS`、`MINERU_OFFICIAL_TIMEOUT_SECONDS` | 否 | 轮询间隔 / 解析超时，示例为 5 / 900 秒 |
| `MONGODB_URI`、`MONGODB_DB` | 是，可沿用本地值 | `mongodb://localhost:27017` / `paper_agent` |
| `MILVUS_HOST`、`MILVUS_PORT` | 是，可沿用本地值 | `localhost` / `19530` |
| `GRAPH_LLM_MODEL`、`GRAPH_LLM_BASE_URL`、`GRAPH_LLM_API_KEY` | 按图谱服务配置 | 示例使用本机 Ollama `qwen3-graph:8b`，独立于普通对话模型 |
| 其余 `GRAPH_*` | 否 | 请求限制、硬超时、租约、重试与熔断，详见示例文件 |
| `USE_MCP`、`MCP_ARXIV_URL` | 否 | 示例启用 arXiv MCP，地址为 `http://localhost:8060/sse` |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | Semantic Scholar 搜索凭据 |
| `UPLOAD_*` | 否 | 批量文件数、大小、待处理上限与任务保留时间 |
| `PA_DATA_ROOT` | 否 | Milvus 及其依赖的数据目录，默认 `./.local-data` |

API 密钥仅保存在本机 `.env`。本地 PDF、上传文件、数据库与缓存不应提交到 Git。

## MinerU 论文解析

Paper Agent 统一使用 MinerU 官方精准 API 解析论文，不再维护本地 MinerU 运行时或 pdfplumber 降级路径。先在 MinerU 平台创建 API Token，再填写本机 `.env`：

```dotenv
MINERU_OFFICIAL_TOKEN=
MINERU_OFFICIAL_BASE_URL=https://mineru.net
MINERU_OFFICIAL_POLL_SECONDS=5
MINERU_OFFICIAL_TIMEOUT_SECONDS=900
```

Token 只填写在本机 `.env`，不要写入 `.env.example` 或提交到 Git。没有配置 Token 时应用会在启动阶段给出明确错误，不会以低精度解析器继续运行。官方 API 失败时，上传队列会自动重试该论文一次；仍失败则清理半成品、记录原因并继续下一篇。上传面板和近 7 天历史批次会显示“官方 VLM · 解析 N.N 秒”，日志会记录上传、等待、下载和总耗时，但不会输出 Token、签名上传地址或结果下载地址。

### MinerU 本地部署参考

MinerU 本身也可以独立部署到本地。官方 Docker 方案适用于 Linux，以及启用了 WSL2 的 Windows；建议先确认 NVIDIA 驱动、CUDA 与显存满足 MinerU 当前版本要求。以下流程只用于部署 MinerU 服务，不是 Paper Agent 的解析模式切换入口。

```bash
# 1. 下载官方 Dockerfile 并构建镜像
curl -L https://raw.githubusercontent.com/opendatalab/MinerU/master/docker/global/Dockerfile -o Dockerfile
docker build -t mineru:latest -f Dockerfile .

# 2. 下载官方 Compose 配置
curl -L https://raw.githubusercontent.com/opendatalab/MinerU/master/docker/compose.yaml -o compose.yaml

# 3. 启动 MinerU Web API
docker compose -f compose.yaml --profile api up -d
```

启动后访问 `http://localhost:8000/docs` 查看接口文档，或请求 `http://localhost:8000/health` 检查服务状态。MinerU Web API 与 Paper Agent 默认都占用主机 `8000` 端口，同时运行时需要修改 `compose.yaml` 的主机端口映射。停止服务可运行：

```bash
docker compose -f compose.yaml --profile api down
```

MinerU 的镜像依赖和启动参数可能随版本变化，实际部署时以 [MinerU 官方 Docker 部署文档](https://github.com/opendatalab/MinerU/blob/master/docs/zh/quick_start/docker_deployment.md) 为准。

## 批量上传队列

选择 PDF 后，文件会先进入可移除的“待上传清单”；可以多次选择后再点击一次“开始上传”，所有接受的文件会成为同一个 MongoDB 批次并严格串行执行。页面会显示“本批上传”的逐篇状态与成功/处理中/失败/跳过统计，并保留近 7 天批次供刷新后查看。论文库只展示已成功入库的论文，不能作为上传成功率的依据。

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

## 研究图谱（Evidence Graph V4）

论文进入 `indexed` 状态后会自动进入持久化图谱队列。工作者从原文 chunks 分批抽取主张，执行 Schema 核验、原文证据定位、Entity Resolution 和 Fact Resolution，最终保存规范实体、主张、事实及其来源。

- 实体别名、缩写和相近表述优先走确定性规则；只有歧义候选才调用 Resolution LLM。
- 同一 Fact 可聚合多篇论文的支持或反对 Claim，每条 Claim 保留 paper、chunk、section/page、原文 evidence 和版本信息。
- 后台任务带租约、心跳、硬超时和阶段检查点；临时限流使用持久化退避，明确的永久配额错误单独处理。
- 图谱可以通过 `GRAPH_LLM_*` 使用独立的本地模型。模型空响应、输出截断和 JSON 协议异常有单独诊断；不能把不完整结果当作成功。
- 页面与 Retriever 继续读取兼容关系投影；图谱用于关联与导航，最终回答的证据仍来自本次检索到的论文 chunks。

图谱页面为 `GET /graph`；主要接口包括 `GET /api/research-graph`、`GET /api/research-graph/status`、`GET /api/research-graph/jobs`、`POST /api/research-graph/jobs/retry` 和 `PATCH /api/research-graph/edges/{edge_id}`。应用启动时会对已索引论文与当前图谱版本做增量对账。

## 仓库结构（Directory Structure）

| 路径 | 内容 |
| --- | --- |
| `agents/`、`graph/`、`state/` | Agent、工作流与状态 |
| `core/` | 依赖装配、会话上下文、搜索准入与证据规则 |
| `knowledge_graph/` | 图谱协议、Schema、实体与事实归一 |
| `storage/`、`tools/` | 数据存储、检索、解析与后台任务 |
| `web/` | FastAPI 服务与前端页面 |
| `tests/`、`evaluation/` | 测试源码与评测工具 |
| `docs/` | 技术架构、运行说明和历史设计文档 |
| `arxiv-mcp/` | 独立 MCP 服务源码与其许可证 |

## 评测与测试

### Eval：用标注问题集观察质量

先完成服务配置并将目标论文入库，再复制示例，为每个问题填写期望命中的 arXiv ID 与是否应拒答：

```powershell
Copy-Item evaluation/cases.example.json evaluation/cases.local.json
python scripts/evaluate.py --cases evaluation/cases.local.json
```

脚本输出逐条结果与 JSON 汇总，指标定义见 [evaluation/metrics.py](../evaluation/metrics.py)。

| 指标 | 统计口径 |
| --- | --- |
| **Recall@5** · `recall_at_5` | 期望论文在前 5 篇去重检索论文中的覆盖率，再对用例取均值；期望集合为空时记为 1 |
| **Citation Pass Rate** · `citation_pass_rate` | 证据规则状态为 `pass` 的用例占比，包含无证据时正确拒答的用例；衡量来源合规，不等同于事实准确率 |
| `abstention_accuracy` | 应拒答用例中，同时满足“无检索论文”和“证据规则通过”的比例；无此类用例时记为 1 |
| `latency_ms_p50` | 工作流端到端耗时的中位数（毫秒） |

使用固定论文库、模型配置与标注集比较改动前后的结果。示例文件提供用例格式，评测数值需在实际入库论文上运行获得。`evaluation/cases.local.json` 与 `evaluation/results/` 已加入 Git 忽略规则。

### 自动化测试与界面验收

测试源码随仓库保存。部分测试需要运行中的 MongoDB、Milvus 或模型依赖，具体范围见 [测试计划](test_plan.md)。先安装 `requirements-dev.txt`，再根据本机环境选择执行。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
docker compose --env-file .env.example config --quiet
python -B -c "import main; from web.app import app; print(len(app.routes))"
```

界面人工验收：切换深浅主题后刷新页面；查看会话和论文库之间主题是否保持；提交一次有证据的问答和一次无证据的问答；确认时间线显示节点耗时、证据面板状态明确、服务失败后发送按钮可再次使用。
