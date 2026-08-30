# Paper Agent 技术架构文档

## 1. 系统概览

Paper Agent 是一个基于 LangGraph 的多 Agent 学术论文助手，支持论文搜索、入库、分析和问答。

```
┌─────────────────────────────────────────────────────────┐
│                    Web 前端 (HTML/JS)                    │
│              http://localhost:8000                       │
└──────────────────────┬──────────────────────────────────┘
                       │ SSE / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI 后端 (web/app.py)                │
│              Session 管理 + Workflow 编排                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LangGraph Workflow (状态图)                 │
│                                                         │
│  ┌──────────┐                                           │
│  │ Supervisor │ ← 路由中枢                              │
│  └─────┬────┘                                           │
│        ├──── direct ──→ DirectAnalyzer ──→ Presenter    │
│        ├──── fetcher ─→ Translator → Fetcher → Presenter│
│        └──── retriever → Retriever → Analyzer → Critic  │
│                                    → Presenter          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    基础设施层                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐ │
│  │ MongoDB  │ │ Milvus  │ │ MinerU  │ │ arXiv API    │ │
│  │ 元数据   │ │ 向量库  │ │ PDF解析 │ │ 论文搜索     │ │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 2. Agent 定义

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **Supervisor** | 路由调度，判断用户意图 | user_query + 对话上下文 | next_agent, search_query, target_paper |
| **Translator** | 中文查询翻译为英文搜索词 | user_query | search_query |
| **Fetcher** | 搜索 arXiv，下载 PDF，解析，分块，向量化，入库 | search_query | target_papers, answer |
| **DirectAnalyzer** | 单篇论文快速分析：下载→解析→全文喂LLM | user_query, target_paper | analysis, answer |
| **Retriever** | 从 Milvus 检索相关 chunks（HybridSearch） | user_query | retrieved_chunks |
| **Analyzer** | 基于检索结果综合分析 | retrieved_chunks, user_query | analysis |
| **Critic** | 评估分析质量，决定是否重试 | analysis, retrieved_chunks | critic_score, next_agent |
| **Presenter** | 格式化最终回复 | analysis/answer | answer |

## 3. 路由逻辑

### 3.1 Supervisor 路由规则

```
用户输入
  │
  ├─ 单篇论文分析 ──────→ direct (DirectAnalyzer)
  │  · 指定了具体论文标题
  │  · "这篇论文"/"它的..." (跟随意图 + 有上下文)
  │
  ├─ 多篇论文入库 ──────→ fetcher (Translator → Fetcher)
  │  · "搜索/找论文/下载"
  │
  ├─ 知识库问答 ────────→ retriever (Retriever → Analyzer)
  │  · "对比/区别/哪些论文"
  │  · 跟随意图但无上下文 (新对话)
  │
  └─ 闲聊/无关 ────────→ END (Presenter)
```

### 3.2 跟随意图检测

```python
# 检测关键词
followup_patterns = ["这篇论文", "该论文", "上一篇", "它的", ...]

# 跟随意图时：
if 有对话上下文 → 提取目标论文标题 → 走 direct
if 无对话上下文 → 走 retriever (让它自己找)
```

## 4. 数据流

### 4.1 单篇论文分析 (DirectAnalyzer)

```
用户: "分析这篇论文 XXX"
  │
  ▼
Supervisor → direct
  │
  ▼
DirectAnalyzer
  │
  ├─ 检查 MongoDB 是否已有
  │   ├─ 有 → 读取 full_text
  │   └─ 无 → arXiv 搜索 → 下载 PDF → MinerU 解析
  │
  ├─ 动态章节提取 (根据用户问题)
  │   · 实验相关 → 提取 method + experiment
  │   · 方法相关 → 提取 method + experiment
  │   · 全面分析 → 提取所有核心章节
  │
  ├─ 全文存 MongoDB (供后续查询)
  │
  ├─ 分块 + Embedding → 存 Milvus (后台入库)
  │
  └─ 核心章节喂 LLM → 分析结果
  │
  ▼
Presenter → 输出
```

### 4.2 多篇论文入库 (Fetcher)

```
用户: "搜索 XXX 论文"
  │
  ▼
Supervisor → fetcher
  │
  ▼
Translator → 英文搜索词
  │
  ▼
Fetcher
  │
  ├─ arXiv 搜索 (MCP 优先，降级直接 API)
  │
  ├─ 逐篇处理 (串行，避免资源爆炸)
  │   ├─ 下载 PDF
  │   ├─ MinerU 解析
  │   ├─ 分块 (2000字符/块，300重叠)
  │   ├─ Embedding (batch=4，清理显存)
  │   └─ 存 MongoDB + Milvus
  │
  └─ 入库结果反馈
  │
  ▼
Presenter → 输出
```

### 4.3 知识库问答 (Retriever)

```
用户: "对比 X 和 Y"
  │
  ▼
Supervisor → retriever
  │
  ▼
Retriever
  │
  ├─ MultiQuery → 生成 3-4 个查询变体
  │
  ├─ 论文级检索 (每个变体)
  │   · Embedding 查询
  │   · 与所有论文的 title+abstract 向量计算余弦相似度
  │   · 取 top-5 论文
  │
  ├─ Chunk 级检索 (限定 top-5 论文)
  │   · Milvus 向量搜索 → top-10 chunks
  │   · BM25 关键词搜索 → top-10 chunks
  │   · RRF 融合 → 最终排序
  │
  ├─ 合并去重所有变体结果
  │
  ├─ Section 过滤 (按意图)
  │
  └─ 返回 top-15 chunks
  │
  ▼
Analyzer → LLM 分析
  │
  ▼
Critic → 评估 (60分+无严重幻觉=pass)
  │
  ▼
Presenter → 输出
```

## 5. 存储架构

### 5.1 MongoDB

```
paper_agent 数据库
  │
  ├── papers 集合
  │   ├── arxiv_id: string (主键)
  │   ├── title: string
  │   ├── abstract: string
  │   ├── authors: list[string]
  │   ├── pdf_url: string
  │   ├── full_text: string (MinerU 解析的全文)
  │   ├── status: "pending" | "parsed" | "chunked" | "indexed"
  │   ├── created_at: datetime
  │   └── updated_at: datetime
  │
  ├── chunks 集合
  │   ├── paper_arxiv_id: string (外键)
  │   ├── chunk_index: int
  │   ├── content: string (分块文本)
  │   └── metadata: {section, page, level, ...}
  │
  ├── sessions 集合
  │   ├── session_id: string
  │   ├── title: string
  │   ├── messages: [{role, content, created_at, timeline}]
  │   ├── created_at: datetime
  │   └── updated_at: datetime
  │
  └── user_memory 集合
      ├── user_id: string
      └── interests: list[string]
```

### 5.2 Milvus

```
paper_chunks Collection
  │
  ├── id: INT64 (主键, auto_id)
  ├── paper_arxiv_id: VARCHAR (论文ID)
  ├── chunk_index: INT64 (分块序号)
  ├── content: VARCHAR(8192) (分块文本)
  ├── embedding: FLOAT_VECTOR(1024) (BGE-m3 向量)
  └── metadata_json: VARCHAR(4096) (元数据JSON)

索引: IVF_FLAT, nlist=128, COSINE 相似度
```

## 6. 检索链路

```
查询
  │
  ▼
MultiQuery (LLM 生成 3-4 变体)
  │
  ▼
每个变体:
  │
  ├─ 论文级检索
  │   · 计算查询向量与每篇论文 title+abstract 的余弦相似度
  │   · 取 top-5 论文
  │
  ├─ Chunk 级检索 (限定 top-5 论文)
  │   ├─ Milvus 向量搜索 → top-10
  │   └─ BM25 关键词搜索 → top-10
  │       └─ RRF 融合 → 最终排序
  │
  └─ 合并到全局结果集

合并去重 → 按 score 排序 → top-15
  │
  ▼
Section 过滤 (按意图: 实验/方法/背景/结论)
  │
  ▼
最终结果 (top-10)
```

## 7. 性能优化

| 优化项 | 措施 | 效果 |
|--------|------|------|
| Embedding 显存 | batch_size=4 + cuda.empty_cache() | 显存峰值 ~6-8GB |
| 论文处理 | 串行 + 2秒间隔 | 避免资源爆炸 |
| 内存回收 | gc.collect() 每篇论文处理后 | 防止内存泄漏 |
| 动态章节 | DirectAnalyzer 按查询提取核心章节 | 70K→15K 字符 |
| MCP 降级 | MCP 失败/空结果 → 直接 arXiv API | 服务可用性 |
| 论文缓存 | MongoDB 存全文，Milvus 存向量 | 避免重复下载 |

## 8. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/chat | 对话入口 (SSE 流式响应) |
| GET | /api/test | 健康检查 |
| POST | /api/upload | 上传 PDF 论文 |
| GET | /api/papers | 论文列表 |
| DELETE | /api/papers/{id} | 删除论文 |
| POST | /api/compare | 多论文对比 |
| GET | /api/citations/{id} | 引用关系 |
| GET | /api/sessions | 会话列表 |
| GET | /api/sessions/{id} | 会话详情 |
| DELETE | /api/sessions/{id} | 删除会话 |
| WS | /ws/{session_id} | WebSocket 实时状态 |

## 9. 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | - |
| Agent 编排 | LangGraph | - |
| LLM | mimo-v2.5 (API) | - |
| Embedding | BAAI/bge-m3 (SentenceTransformer) | dim=1024 |
| 向量库 | Milvus | IVF_FLAT |
| 文档数据库 | MongoDB | - |
| PDF 解析 | MinerU 官方精准 API（VLM） | - |
| 论文搜索 | arXiv MCP / arXiv Direct API | - |
| GPU | NVIDIA RTX 5070 Ti (16GB) | CUDA 12.8 |

## 10. 启动命令

```bash
# Web 服务
HF_HUB_DISABLE_TRANSFORMERS_SAFE_LOAD_CHECK=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000

# 或使用 start.bat (Windows)
```

## 11. 环境变量

```env
# LLM
LLM_MODEL=mimo-v2.5
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_API_KEY=your_key

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=  # 留空自动检测

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=paper_agent

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# MinerU 官方精准 API（必填）
MINERU_OFFICIAL_TOKEN=your_token
MINERU_OFFICIAL_BASE_URL=https://mineru.net
MINERU_OFFICIAL_POLL_SECONDS=5
MINERU_OFFICIAL_TIMEOUT_SECONDS=900

# MCP (可选)
USE_MCP=true
MCP_ARXIV_URL=http://localhost:8050/sse
```
