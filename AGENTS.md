# Paper Agent - Agent 工作手册

## 项目概述

Paper Agent 是一个基于多 Agent 协作的学术论文智能助手。支持论文搜索、分析、记忆和跨论文关联。

## 架构设计

### Fast/Slow Path

```
Fast Path (99%): 不调用 LLM
────────────────────────────────
PDF → Chunk → Embedding → Milvus
                           ↓
                      BM25 Index
                           ↓
                      Hybrid Search (BM25 + Vector)
                           ↓
                      Reranker (Cross-Encoder)
                           ↓
                      Weight Update (规则)

Slow Path (1%): 调用 LLM
────────────────────────────────
Section Summary      ← LLM
Paper Summary        ← LLM
Concept Extraction   ← LLM
Memory Merge         ← 条件LLM
Reflection           ← LLM
Final Answer         ← LLM
```

### Agent 架构

```
START → Supervisor → Translator → Fetcher → Presenter → Reflector → END
        Supervisor → Retriever → Analyzer → Critic → Presenter → Reflector → END
```

## Agent 职责

### Supervisor
- **职责**：路由调度，判断用户意图
- **输入**：用户原始查询
- **输出**：`{next_agent, search_query, reason}`
- **规则**：
  - 搜索/找论文 → fetcher
  - 提问/分析 → retriever
  - 闲聊 → presenter

### Translator
- **职责**：将用户输入翻译为英文搜索词
- **输入**：`user_query`
- **输出**：`{search_query}`

### Fetcher
- **职责**：从 arXiv/Semantic Scholar 搜索论文，解析 PDF，分块入库
- **输入**：`search_query`
- **输出**：`{target_papers, answer}`
- **流程**：
  1. 搜索论文（Semantic Scholar 优先）
  2. 下载 PDF → 解析（MinerU 优先）
  3. 分块 → Embedding → 存入 Milvus
  4. 初始化 Paper Memory + Section Memory

### Retriever
- **职责**：语义检索相关论文片段
- **输入**：`user_query`
- **输出**：`{retrieved_chunks}`
- **流程**：
  1. MultiQuery（LLM 生成 3-5 个变体）
  2. 两层检索：论文级 → Chunk级
  3. Section-aware 过滤
  4. Hybrid Search（BM25 + Vector）
  5. Reranker 重排序

### Analyzer
- **职责**：综合分析检索到的论文内容
- **输入**：`retrieved_chunks`, `user_query`
- **输出**：`{analysis}`
- **规则**：
  - 严格基于检索内容，不编造
  - 每个论点标注来源论文
  - 使用 ReAct 模式，可调用工具查询更多内容

### Critic
- **职责**：评估回答质量，决定是否重试
- **输入**：`analysis`, `retrieved_chunks`
- **输出**：`{next_agent, iteration}`
- **规则**：
  - score >= 70 且无幻觉 → pass
  - 否则 → revise（重试）

### Presenter
- **职责**：格式化最终回复
- **输入**：`analysis`, `retrieved_chunks`
- **输出**：`{answer}`
- **规则**：
  - 第一句话直接回答问题
  - 引用论文用 [论文标题] 格式

### Reflector
- **职责**：生成反思记忆（洞察、问题、未来方向）
- **输入**：`analysis`, `retrieved_chunks`
- **输出**：`{reflection}`
- **存储**：Reflection Memory

### Summarizer
- **职责**：生成 Section/Paper 摘要
- **调用时机**：论文入库后（Slow Path）
- **输出**：Section Summary + Paper Summary

### ConceptExtractor
- **职责**：从论文中提取概念
- **调用时机**：论文入库后（Slow Path）
- **输出**：概念列表（name, definition, aliases）

### ImportanceEvaluator
- **职责**：评估概念重要性（规则，不用 LLM）
- **公式**：`score = 0.4*frequency + 0.3*citation + 0.2*access + 0.1*recency`

## Memory 体系

```
┌─────────────────────────────────────────────────┐
│ Working Memory (LangGraph State)                │
├─────────────────────────────────────────────────┤
│ Session Memory (MongoDB)                        │
├─────────────────────────────────────────────────┤
│ Paper Memory (MongoDB)                          │
│   - title, authors, sections, contributions     │
│   - datasets, metrics, keywords                 │
├─────────────────────────────────────────────────┤
│ Section Memory (MongoDB + Milvus)               │
│   - heading, summary, chunk_count               │
├─────────────────────────────────────────────────┤
│ Concept Memory (MongoDB + Milvus)               │
│   - name, definition, source_papers             │
├─────────────────────────────────────────────────┤
│ Knowledge Graph (Neo4j 或内存)                  │
│   - Paper → proposes → Concept                  │
│   - Concept → used_by → Paper                   │
├─────────────────────────────────────────────────┤
│ Reflection Memory (MongoDB)                     │
│   - insights, questions, future_directions       │
├─────────────────────────────────────────────────┤
│ User Memory (MongoDB)                           │
│   - interests, focus, patterns, entities        │
│   - 生命周期: 形成→强化→合并→衰减→遗忘          │
└─────────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | MiMo v2.5 | 对话、分析、摘要、概念提取 |
| Embedding | BGE-M3 | 文本向量化 |
| Reranker | BGE-Reranker-v2-M3 | 重排序 |
| 向量数据库 | Milvus | 语义检索 |
| 元数据库 | MongoDB | Memory 存储 |
| 知识图谱 | Neo4j（可选） | 实体关系 |
| PDF 解析 | MinerU / pdfplumber | PDF → Markdown |
| 框架 | LangGraph | Agent 编排 |
| Web | FastAPI + SSE | 流式输出 |

## 检索架构

```
Query → MultiQuery (LLM)
         ↓
    Section Retrieval
         ↓
    Chunk Retrieval
         ↓
    Hybrid Search (BM25 + Vector RRF)
         ↓
    Reranker (Cross-Encoder)
         ↓
    Top-K Results
```

## LLM 调用统计

### 每篇论文（入库时）
- Section Summary: 5-10 次
- Paper Summary: 1 次
- Concept Extraction: 5-10 次
- **总计: 12-22 次**

### 每次查询
- Supervisor: 1 次
- MultiQuery: 1 次
- Analyzer: 1-3 次
- Critic: 1 次
- Presenter: 1 次
- Reflector: 1 次
- **总计: 6-9 次**

## 状态定义 (AgentState)

```python
{
    "user_id": str,              # 用户ID（用于记忆）
    "user_query": str,           # 用户原始查询
    "search_query": str,         # 提取的英文搜索词
    "target_papers": list,       # 搜索到的论文列表
    "retrieved_chunks": list,    # 检索到的相关 chunks
    "analysis": str,             # Analyzer 的分析结论
    "answer": str,               # 最终回复
    "reflection": dict,          # 反思记忆
    "next_agent": str,           # 路由决策
    "iteration": int,            # 当前迭代轮次
    "max_iterations": int,       # 最大重试次数
    "session_id": str,           # 会话ID
    "error": str,                # 错误信息
}
```

## 错误处理

- Agent 失败 → 返回错误状态，不崩溃
- 路由函数检查非法路由 → 强制终止
- Workflow 超时 → 300 秒超时限制
- Critic 异常 → 默认 pass，继续执行
- LLM 调用失败 → 重试 2 次
