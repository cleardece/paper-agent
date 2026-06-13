# Paper Agent - Agent 工作手册

## 项目概述

Paper Agent 是一个基于多 Agent 协作的学术论文智能助手。用户可以通过自然语言对话搜索、获取和分析学术论文。

## Agent 架构

```
用户输入 → Supervisor → 路由决策
                        ├─→ Fetcher → 搜索入库
                        ├─→ Retriever → Analyzer → Critic → Presenter → 回答
                        └─→ Presenter → 闲聊回复
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

### Fetcher
- **职责**：从 arXiv/Semantic Scholar 搜索论文，解析 PDF，分块入库
- **输入**：`search_query`（英文关键词）
- **输出**：`{target_papers, next_agent}`
- **规则**：
  - 搜索结果缓存 1 小时
  - 跳过已入库的论文
  - PDF 解析优先用 MinerU，fallback 到 pdfplumber

### Retriever
- **职责**：语义检索相关论文片段
- **输入**：`user_query`
- **输出**：`{retrieved_chunks}`
- **规则**：
  - Embedding 向量检索 top_k=10
  - 返回 chunks 包含论文标题、内容、分数

### Analyzer
- **职责**：综合分析检索到的论文内容
- **输入**：`retrieved_chunks`, `user_query`
- **输出**：`{analysis}`
- **规则**：
  - 基于论文内容回答，不要编造
  - 明确引用来源
  - 上下文窗口限制 50K 字符
  - 使用 ReAct 模式，可调用工具查询更多内容

### Critic
- **职责**：评估回答质量，决定是否重试
- **输入**：`analysis`, `retrieved_chunks`
- **输出**：`{critic_score, next_agent, iteration}`
- **规则**：
  - 5 维度评分（faithfulness, relevancy, completeness, citation_accuracy, coherence）
  - score >= 70 且无幻觉 → pass
  - 否则 → revise（重试 Retriever）
  - 最多重试 max_iterations 次

### Presenter
- **职责**：格式化最终回复
- **输入**：`analysis`, `retrieved_chunks`
- **输出**：`{answer}`
- **规则**：
  - 第一句话直接回答问题
  - 引用论文时用 [论文标题] 格式
  - 总字数 500-1500 字
  - 如果是闲聊，简短回复

## 状态定义 (AgentState)

```python
{
    "user_query": str,           # 用户原始查询
    "search_query": str,         # 提取的英文搜索词
    "target_papers": list,       # 搜索到的论文列表
    "retrieved_chunks": list,    # 检索到的相关 chunks
    "analysis": str,             # Analyzer 的分析结论
    "answer": str,               # 最终回复
    "critic_score": dict,        # 质量评分
    "next_agent": str,           # 路由决策
    "iteration": int,            # 当前迭代轮次
    "max_iterations": int,       # 最大重试次数（默认 2）
    "error": str,                # 错误信息
}
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | MiMo v2.5 | 对话、分析、评估 |
| Embedding | BGE-M3 (568M) | 文本向量化 |
| 向量数据库 | Milvus | 语义检索 |
| 元数据库 | MongoDB | 论文元数据、分块内容 |
| PDF 解析 | MinerU / pdfplumber | PDF → Markdown |
| 框架 | LangGraph | Agent 编排 |
| Web | FastAPI + SSE | 流式输出 |

## 缓存策略

- **会话级缓存**：每个 Session 独立缓存
- **缓存内容**：搜索结果、Embedding 向量、论文解析结果
- **TTL**：会话生命周期内有效
- **删除会话时**：连带删除所有缓存

## 上下文窗口

| Agent | 限制 | 说明 |
|-------|------|------|
| Analyzer | 50K 字符 | MiMo 32K token，留足空间 |
| Retriever | top_k=10 | 检索最多 10 个 chunks |
| Critic | 无限制 | 只评估 Analyzer 输出 |

## 错误处理

- Agent 失败 → 返回错误状态，不崩溃
- 路由函数检查非法路由 → 强制终止
- Workflow 超时 → 300 秒超时限制
- Critic 异常 → 默认 pass，继续执行
