# Paper Agent 技术架构

## 1. 系统概览

Paper Agent 是一个基于 FastAPI 和 LangGraph 的学术论文助手。系统将论文搜索与入库、基于证据的问答、多轮论文上下文和可审核研究图谱分成独立链路，通过 MongoDB 与 Milvus 共享稳定的论文 ID。

```text
浏览器（对话 / 论文库 / 研究图谱）
        |
        | HTTP + SSE / WebSocket
        v
FastAPI (web/app.py)
        |
        +-- LangGraph 问答工作流
        |     PaperContextResolver -> Supervisor -> TurnContext
        |       +-- Fetcher
        |       +-- DirectAnalyzer
        |       +-- Retriever -> Analyzer -> Critic -> Presenter
        |
        +-- 持久化上传队列 -> MinerU -> Chunk -> MongoDB / Milvus
        |
        +-- 可恢复图谱队列 -> Evidence Graph V4
              -> Entity / Claim / Fact / Provenance
```

## 2. 对话工作流

| 节点 | 职责 |
| --- | --- |
| `PaperContextResolver` | 在路由前用显式目标、arXiv ID/DOI、本地标题、会话焦点和结构化历史解析论文 ID |
| `Supervisor` | 只判断 `analyze` / `rag` / `compare` / `search` / `general` 动作，不再猜论文标题 |
| `TurnContext` | 将查询、意图、主论文、参与论文和外部搜索权限投影给下游 Agent |
| `Fetcher` | 唯一允许执行外部论文发现的 Agent；只接收经验证的 `SearchRequest` |
| `DirectAnalyzer` | 分析已解析的本地单篇论文；失败时不把原问题转发给 arXiv |
| `Retriever` | 在已入库论文中执行论文级筛选、BM25 + 向量检索、RRF 融合和重排 |
| `Analyzer` | 仅基于本次 `retrieved_chunks` 组织分析 |
| `Critic` | 检查证据归属、回答质量和是否需要重检索/修订 |
| `Presenter` | 输出面向用户的答案、状态和引用 |

```text
用户消息
  -> 解析 PaperContext
  -> 判断动作意图
       analyze + primary_paper_id -> DirectAnalyzer
       compare/rag                -> Retriever -> Analyzer
       search                     -> SearchAdmissionGate -> Fetcher
       general                    -> Presenter
```

外部搜索只对明确的 `search` 意图开放。`SearchQueryBuilder` 只构建 arXiv ID、标题或简短关键词请求，不把整段分析问题原样发给 arXiv。没有唯一论文上下文的分析请求返回稳定业务错误，不会隐式搜索。

## 3. 多轮论文上下文

`PaperContextResolver` 按以下顺序决定论文身份：

1. 前端显式传入的 `target_paper_id`；
2. 当前消息中的本地 ID、arXiv ID 或 DOI；
3. 本地论文库的标题匹配或明确切换表达；
4. 会话中的唯一 `primary_paper_id`；
5. 近期消息保存的结构化 `paper_context`；
6. 返回 unresolved / ambiguous，不从助手文本里猜标题。

只要当前会话有唯一主论文，“总结实验结果”这类不带代词的追问也会继承该论文。Session 保存 `paper_focus`，每条消息可保存当轮 `paper_context`。`SessionStateReducer` 是唯一更新会话焦点的组件；失败或未解析轮次不覆盖已有焦点。

## 4. 论文入库与检索

```text
上传 PDF / Fetcher 获取论文
  -> 持久化串行队列
  -> MinerU 官方 VLM 解析
  -> 分块 -> MongoDB
  -> BGE-M3 Embedding -> Milvus
  -> 论文状态 indexed
```

上传批次和单篇任务持久化到 MongoDB，服务重启后恢复未完成工作。检索链路先缩小论文范围，再在目标 chunks 中执行 BM25 与向量检索，通过 RRF 融合和 Cross-Encoder 重排返回证据。论文特定问答使用 `TurnContext.paper_ids` 限定范围。

## 5. Evidence Graph V4

论文进入 `indexed` 后由上传 Worker 唤醒图谱 Worker。应用启动时还会对已索引论文和当前 `evidence-graph-v4` 版本做增量对账。

```text
Paper chunks
  -> 分批 Claim 抽取
  -> 分批核验 + 固定 Schema 映射
  -> 原文 evidence 定位
  -> Entity Resolution
       exact canonical -> exact alias -> vector Top-K -> rule score
       -> merge / new / 歧义批量 LLM
  -> Fact Resolution
       exact signature -> vector Top-K -> structural score
       -> existing / new / 歧义批量 LLM
  -> 按论文原子替换系统生成 Claim
  -> 生成兼容 research_graph_edges 投影
```

核心 Schema 使用有限 Entity Type 和 Canonical Predicate。无法可靠映射的关系不进入可用 Fact。同一 Fact 可聚合多条 `support` / `contradict` Claim，每条 Claim 保留评审状态和完整 Provenance。

图谱任务使用租约、心跳、独立子进程硬超时、一次可恢复重试和熔断。LLM 配额耗尽会标记为 `llm_quota_exhausted` 并终止自动重试；配额恢复后可在图谱页面手动重试。

`research_graph_edges` 是页面和 Retriever 的兼容投影，用于论文关联、导航和候选扩展，不代替问答证据。最终回答仍必须基于当前检索到的原文 chunks。

## 6. 存储模型

### MongoDB

| 数据 | 主要内容 |
| --- | --- |
| `papers` / `chunks` | 论文元数据、MinerU 全文、分块、入库与图谱状态 |
| sessions / messages | 对话、Agent 时间线、`paper_focus`、每轮 `paper_context` |
| upload batches / jobs | 持久化上传批次和单篇任务 |
| `research_graph_jobs` | 图谱版本、运行次数、尝试历史、租约、分批进度和诊断 |
| `research_graph_entities` / `research_graph_aliases` | 规范实体、别名与阻断键 |
| `research_graph_claims` / `research_graph_facts` | 带 Provenance 的论文主张与跨论文事实 |
| `research_graph_resolution_cache` | Entity/Fact Resolution 的稳定决策缓存 |
| `research_graph_nodes` / `research_graph_edges` | 页面和旧调用方的兼容投影 |

### Milvus

- 论文/chunk 向量服务于 RAG 论文级与片段级检索。
- `kg_entity_embeddings` 按 Entity Type 检索规范实体 Top-K 候选。
- `kg_fact_embeddings` 按 Predicate 检索规范 Fact Top-K 候选。

KG 向量仅用于候选召回，不单独决定合并。Milvus KG collection 不可用时，解析器使用有界 MongoDB 候选继续工作。

## 7. 主要 HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/chat` | SSE 对话入口 |
| `POST` | `/api/uploads` / `/api/upload` | 批量上传与单文件兼容入口 |
| `GET` | `/api/upload-batches` / `/api/upload-batches/{batch_id}` | 上传批次列表与详情 |
| `GET` / `DELETE` | `/api/papers` / `/api/papers/{id}` | 论文列表与删除 |
| `POST` | `/api/compare` | 多论文对比 |
| `GET` | `/api/sessions` / `/api/sessions/{id}` | 会话列表与详情 |
| `GET` | `/api/research-profile/{user_id}` | 研究者档案 |
| `GET` | `/api/research-graph` / `/api/research-graph/paper-links` | 证据关系与论文关联 |
| `GET` | `/api/research-graph/status` / `/api/research-graph/jobs` | 图谱覆盖率、任务和尝试历史 |
| `POST` | `/api/research-graph/jobs/retry` | 手动重试失败论文 |
| `PATCH` | `/api/research-graph/edges/{edge_id}` | 确认或拒绝待复核关系 |
| `WS` | `/ws/{session_id}` | 实时任务状态 |

## 8. 故障隔离与恢复

- Agent 返回结构化业务错误，不让单节点异常直接崩溃整个会话。
- 上传任务与图谱任务均持久化；服务重启时回收中断任务或过期租约。
- 图谱抽取子进程有硬超时，主 Worker 持续心跳，不让单篇论文阻塞队列。
- 图谱核验 LLM 漏回单个候选时按项降级为 `needs_review`；只有整个响应无法解析时才拆批重试。
- 证据不在原 chunk 中、Schema 非法或核验拒绝的 Claim 不写入可用 Fact。

## 9. 配置与不变式

环境变量以 `.env.example` 为准：LLM 使用 `LLM_*`，存储使用 `MONGODB_*` / `MILVUS_*`，解析使用 `MINERU_OFFICIAL_*`，上传队列使用 `UPLOAD_*`，图谱队列使用 `GRAPH_*`，外部搜索使用 `USE_MCP`、`MCP_ARXIV_URL` 和 `SEMANTIC_SCHOLAR_API_KEY`。

1. 论文身份在意图路由前解析，下游 Agent 不各自猜测。
2. 非 `search` 意图不允许调用外部论文搜索。
3. 成功的单篇分析必须返回稳定主论文 ID。
4. 最终回答的引用必须来自本次检索证据，图谱不直接充当答案来源。
5. 每个可用 Claim 必须保留可定位的原文证据和版本信息。
6. 新论文只增量处理自身数据并检索有界 Top-K 候选，不触发全库重建。
