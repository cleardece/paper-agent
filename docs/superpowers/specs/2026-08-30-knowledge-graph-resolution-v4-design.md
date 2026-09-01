# Knowledge Graph Resolution V4 设计

## 目标与边界

本次只升级 Research Graph / Knowledge Graph 内部管线，不修改现有 RAG、PDF 解析、Agent 工作流、HTTP 路由和页面交互。现有 `research_graph_edges` 继续作为兼容投影供页面与 Retriever 使用；新的 Canonical Entity、Claim、Fact 与 Provenance 数据存入独立 MongoDB collection。

图谱版本升级为 `evidence-graph-v4`。新论文只处理自身的 mention、claim 和 provenance，并与已有 canonical 数据做 Top-K 对齐；不扫描或重建整个论文库。V3 论文通过现有版本对账机制低优先级增量重建。

## 固定 Schema

Entity Type 固定为：`method`、`model`、`algorithm`、`dataset`、`problem`、`task`、`equation`、`software`、`metric`、`domain`。

Canonical Predicate 固定为：`USES_METHOD`、`USES_MODEL`、`USES_DATASET`、`USES_SOFTWARE`、`SOLVES`、`EVALUATED_BY`、`IMPROVES`、`OUTPERFORMS`、`COMPARES_WITH`、`EXTENDS`、`BASED_ON`、`UNKNOWN`。

LLM 输出中的自由关系只能在 verification 阶段映射到上述集合。无法可靠映射时使用 `UNKNOWN`，且不写入可用 Fact。

## 数据模型

四类核心对象：

- Paper：沿用 `papers` collection。
- Canonical Entity：保存稳定 `entity_id`、canonical name、type、aliases、normalized aliases、domain/context 摘要、embedding 版本和创建/更新时间。
- Claim：保存论文原始陈述、canonical subject/object、predicate、qualifiers、stance、confidence、review status、paper/chunk/section/page/evidence、抽取与验证版本。
- Fact：保存 canonical subject、predicate、object、稳定 signature、支持/反对 claim 计数和聚合状态。

Claim 的 `stance` 固定为 `support` 或 `contradict`。冲突 Claim 指向同一个高层 Fact，不互相覆盖。Fact signature 为 `subject_entity_id|predicate|object_entity_id`；qualifiers 保留在 Claim 中，不进入第一版高层 signature。

## 管线

```text
Paper chunks
  -> batched extraction (Raw Claims)
  -> batched verification + schema normalization
  -> deterministic evidence validation
  -> Entity Resolution
       exact canonical -> exact alias -> vector Top-K -> rule score
       -> merge / new / batched LLM ambiguity
  -> Fact Resolution
       exact signature -> vector Top-K -> structural score
       -> existing / new / batched LLM ambiguity
  -> provenance validation
  -> atomic per-paper graph write
  -> compatibility edge projection
```

抽取和 verification 仍维持每批两次基础 LLM 调用。Normalization 合并进 verification，不增加固定调用。Entity/Fact Resolution 仅在规则无法确定时各最多增加一次 batch LLM 调用。

## Entity Resolution

字符串 normalization 统一 Unicode、大小写、连字符、空白、常见英文复数和 acronym。Canonical name exact match 与 alias exact match 均走 MongoDB 索引，不调用 LLM。

非 exact mention 批量生成 BGE-M3 embedding，并在独立 `kg_entity_embeddings` collection 检索同类型 Top-K。规则评分综合 normalized name、acronym、embedding、type、domain/context 与 relation context：高分自动 merge，低分创建新 Entity，中间区间进入一次 batch LLM。LLM 漏答或输出非法时不让论文任务失败，而是保守创建新 Entity 并标记 unresolved，避免错误 merge。

成功 merge 后把新 mention 追加到 canonical aliases；以后 alias exact match 直接命中 Fast Path。Resolution cache 记录 mention/type/context 到 entity 的决定与 resolver version。

## Fact Resolution

Entity Resolution 完成后才构建 Fact。Exact signature 直接命中。没有 exact match 时，从独立 `kg_fact_embeddings` Top-K 候选中按 canonical subject/predicate/object、qualifiers 与上下文评分。明确相同则 merge，明确不同则新建，模糊项一次 batch LLM。

Embedding 只用于召回，不单独决定 merge。LLM 漏答时保守新建并标记 unresolved。支持与反对证据通过 Claim stance 聚合到同一个 Fact。

## 核验缺失判定的故障处理

当前 V3 把“LLM 未完整返回每个候选”当作整个批次失败。V4 改为逐项容错：合法判定照常使用；缺失、重复冲突、越界或 schema 非法的判定降级为 `uncertain`，保留原 predicate 并记录 `missing_or_invalid_decision`。只有 response 完全无法解析为 JSON 数组时才触发批次重试。

## Provenance 与写入一致性

每个 Claim 必须拥有 paper id、chunk index、section/page、逐字 evidence、完整 chunk context/hash、confidence、extractor/verifier/resolver/graph version。写入前再次检查 evidence 可在原 chunk 中定位。

每篇论文的系统生成 Claim 与兼容 edges 使用同一 processing run id。新结果全部解析和 resolution 成功后再替换该论文旧的系统数据；人工 confirmed/rejected 状态继续保留并映射到对应 Claim。Fact 的计数根据受影响 Fact 增量重算，不全库扫描。

## 兼容性

现有 `search()`、`paper_links()`、`find_related_paper_ids()`、review API 和图谱页面继续读取 `research_graph_edges`。每个 edge 增加 claim/fact/entity/provenance 字段，但保留原有 `relation`、`target_type`、`target_name`、evidence 与 review status。Canonical predicate 映射为现有页面可显示的 legacy relation，不要求修改接口调用方。

## 验收

- verification 漏回候选时，任务继续且该候选进入 `needs_review`，不再出现“没有完整返回”批次失败。
- PINN/PINNs/Physics-Informed Neural Networks 能通过 exact/alias/acronym/vector 路径归一到同一 Entity。
- 相同 subject/predicate/object 的不同论文 Claim 指向同一 Fact；qualifiers 与 provenance 分别保留。
- contradict Claim 不覆盖 support Claim，Fact 可统计双方证据。
- exact/alias/signature 命中不调用 Resolution LLM；歧义项按 Entity/Fact 各一次 batch 调用。
- 新论文只读取 Top-K canonical 候选与自身 chunks，不遍历全库。
- 现有 Research Graph API、页面和 Retriever 调用保持可用。
