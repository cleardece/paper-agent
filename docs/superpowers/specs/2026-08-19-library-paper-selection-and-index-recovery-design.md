# 论文库精确选择与索引恢复设计

## 目标

修复论文库中“分析”操作可能错选论文、把已有 chunks 误判为全文缺失、以及 `chunked` 论文无法自动补齐向量索引的问题。系统必须以数据准确性优先，不得因标题中的单个英文词猜测另一篇论文。

## 已确认的问题

- 论文库入口当前只传标题文本。Supervisor 在没有对话上下文时会选取最近入库论文作为目标。
- DirectAnalyzer 使用“任意一个英文词匹配标题”的规则，能把 MMC 论文错误匹配为包含 `Test` 的其他论文。
- 本地上传路径保存 chunks 和状态，但不冗余保存 `full_text`。因此 `chunked`/`indexed` 是可分析的数据，却被 DirectAnalyzer 误判为全文缺失并下载 PDF。
- DirectAnalyzer 中已有补向量化意图，但被全文缺失分支提前截断，无法从 `chunked` 恢复为 `indexed`。

## 设计

### 1. 论文库选择必须传递稳定 ID

论文库的“分析”操作保存并发送 `arxiv_id`，而不仅是标题。`ChatRequest`、`AgentState` 和 `DirectAnalyzer` 传递 `target_paper_id`。只要该字段存在：

1. Supervisor 跳过标题猜测并路由至 `direct`。
2. DirectAnalyzer 用 `get_paper(target_paper_id)` 精确读取同一篇论文。
3. ID 不存在时返回明确错误，不回退到其他论文或重新搜索。

### 2. 无稳定 ID 时禁止宽松猜测

没有 `target_paper_id` 的普通对话保留标题/上下文能力，但：

- 没有会话上下文时，不再把最近入库论文当作“这篇论文”。
- 标题查找只接受规范化标题相等，或明确引用的完整标题；单个关键词命中不再返回论文。
- 未能精确锁定时，提示用户选择论文或改走检索，不下载一篇猜测的论文。

### 3. chunks 是可分析的权威内容

`full_text` 只是一种可选缓存，而不是论文可用性的判断标准。对于已有论文：

1. 若 `full_text` 存在，使用它。
2. 否则读取按 `chunk_index` 排序的 MongoDB chunks，利用 chunk 的 `metadata.section` 和 `content` 重建分析文本。
3. chunks 为空才视为内容缺失；此时才允许走下载/解析流程。

不会为已有论文重复写入 `full_text`，避免两份正文数据失去一致性。

### 4. chunked 状态自动恢复索引

当精确选中的论文存在 chunks 但状态不是 `indexed` 时：

1. 直接对已有 chunks 生成 embedding，写入 Milvus。
2. 写入成功才更新状态为 `indexed`。
3. embedding 失败时设为 `embedding_failed`；Milvus 写入失败时设为 `milvus_failed`，保留 chunks 以便重试。
4. 单篇分析仍使用已有 chunks；索引恢复失败不会触发 PDF 重下载，也不会把状态伪装为 `indexed`。

## 验收标准

- 点击 MMC 论文的“分析”后，日志和回答引用同一个 `arxiv_id`，不会选择 Ultra Fast Silicon 论文。
- 已有 53 个 chunks 的 MMC 论文不触发 PDF 下载。
- `chunked` 论文在补索引成功后变为 `indexed`；失败时保留 chunks 和明确失败状态。
- 手动模糊提问不再因为一个公共英文词关联到无关论文。
- 现有 `indexed` 论文仍可正常直接分析。
