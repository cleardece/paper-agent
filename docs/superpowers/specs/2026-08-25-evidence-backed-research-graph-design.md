# Evidence-Backed Research Graph Design

## Goal

Add a persistent, source-traceable research graph that helps the existing Hybrid RAG answer cross-paper questions without delaying a paper becoming searchable or allowing unverified LLM relationships into answers.

## First-version scope

The first version builds only these directed relationships, always from one local paper to one normalized entity:

```text
Paper --proposes--> Method
Paper --uses--> Dataset
Paper --uses--> Metric
Paper --compares_with--> Method
```

Every edge requires a source chunk already stored for the paper. The graph does not infer paper-to-paper claims directly; cross-paper connections arise when multiple local papers connect to the same method, dataset, or metric. Citation graphs, research gaps, user-profile nodes, temporal claims, community summaries, and full GraphRAG global search are out of scope for this version.

## Storage and identity

Use MongoDB rather than adding Neo4j. The project already depends on MongoDB, expected local graph size is a research library rather than a web-scale graph, and this keeps deployment portable for users with modest hardware. The existing in-memory/optional-Neo4j `KnowledgeGraph` class is not used as the production source of truth.

Create two collections:

```text
research_graph_nodes
research_graph_edges
```

Nodes have stable IDs:

```json
{
  "node_id": "paper:local_2201.02491v3_5aca582e",
  "node_type": "paper | method | dataset | metric",
  "name": "Moving Morphable Component method",
  "normalized_name": "moving morphable component method",
  "aliases": ["MMC"],
  "schema_version": 1,
  "created_at": "...",
  "updated_at": "..."
}
```

An edge is idempotent per source paper, relation type, target node and evidence chunk:

```json
{
  "edge_id": "sha256(...)",
  "source_node_id": "paper:...",
  "target_node_id": "method:moving-morphable-component-method",
  "relation_type": "proposes | uses | compares_with",
  "source_paper_id": "local_2201.02491v3_5aca582e",
  "evidence": {
    "chunk_index": 12,
    "section": "Method",
    "page": 4,
    "content_hash": "sha256(...)"
  },
  "confidence": 0.0,
  "extractor_version": "research-graph-v1",
  "review_status": "auto | confirmed | rejected",
  "created_at": "...",
  "updated_at": "..."
}
```

The graph repository creates unique indexes for `node_id` and `edge_id`, plus lookup indexes for source/target node, source paper, relation type, and review status.

## Upload and graph-job flow

The established upload worker remains the sole owner of PDF parsing, chunking, embedding, and Milvus insertion:

```text
queued → parsing → chunking → indexing → completed
```

Immediately after a successful `indexed` paper write, it creates a durable `research_graph_jobs` record with status `pending`. The upload job is then marked completed as it is today; the paper is usable through RAG before graph work begins.

A separate, single graph worker runs only while the upload queue has no pending or processing PDF job. A new upload always has priority. This means graph construction does not overlap MinerU parsing and cannot extend the user's upload queue latency. It uses only LLM calls and MongoDB; it does not parse the PDF again or rerun BGE-M3 embeddings.

Graph job statuses are:

```text
pending → extracting → validating → completed
                           ↘ failed
```

Jobs retry once after a transient extraction or database failure. A final graph failure records a safe error and leaves the indexed paper, chunks, and vectors untouched. Rebuilding one paper first removes only its `review_status=auto` edges and orphaned automatic entity nodes attributable solely to that paper; user-confirmed edges are never deleted automatically.

## Evidence-constrained extraction

The graph worker reads only high-signal chunks whose normalized section heading matches abstract, introduction, method, experiment, result, evaluation, conclusion, or limitation. It limits input to a configurable maximum of twelve chunks and sends them to the existing LLM with `temperature=0` and a strict JSON schema.

The LLM may emit candidates in this form:

```json
{
  "relations": [{
    "relation_type": "uses",
    "entity_name": "MBB beam benchmark",
    "entity_type": "dataset",
    "aliases": ["MBB beam"],
    "evidence_chunk_index": 23,
    "confidence": 0.87
  }]
}
```

The extractor does not choose arbitrary source entities: every candidate source is the current paper. The validator rejects a candidate when the relation or entity type is not whitelisted, the evidence chunk is not in the supplied set, the entity name is empty/generic/overlong, the confidence is outside `0..1`, or the candidate's normalized evidence does not contain a non-trivial token from the entity name or alias. It writes only validated candidates.

Concept canonicalization is deterministic for this version: Unicode normalization, lower-casing, whitespace/punctuation collapse, and aliases from the extractor. It does not use LLM-based cross-paper entity merging; ambiguous names remain separate until an explicit future review workflow.

## Retrieval and answer behavior

Existing single-paper and ordinary semantic queries continue to use the current Hybrid RAG path unchanged.

For queries that request relationships across papers, such as “哪些论文使用同一数据集” or “MMC 方法在哪些论文中被比较过”, the supervisor adds graph retrieval before normal chunk retrieval:

```text
user question
  → graph entity/edge lookup
  → local paper IDs and evidence chunk IDs
  → existing Milvus/BM25 retrieval restricted to those papers
  → current Analyzer/Presenter answer with original-chunk citations
```

Graph edges rank and constrain candidates; they never become the final source of truth. If graph lookup returns no validated edge or the follow-up chunk retrieval cannot reproduce evidence, the system falls back to ordinary RAG and states that no graph-supported relation was found.

## User review and interface

Add a `研究图谱` entry next to the paper library. The first interface is an evidence-first relation explorer, not an unconstrained network drawing:

- search a paper or entity;
- filter by method, dataset, metric, or relation type;
- show connected local papers and relation cards;
- expand every relation card to its source paper, section, page, and exact stored chunk;
- mark one relationship confirmed or rejected, with rejection removing it from graph-assisted retrieval;
- show per-paper graph status: pending, extracting, ready, failed, or retryable.

The interface may add a compact focus graph after the relation explorer is stable, but no third-party visualization dependency is required for the first version.

## Evaluation and upgrade gates

The graph is not expanded based on graph size. Before a new relation family is enabled:

1. Sample 30–50 automatic edges from the relevant type and manually label correct, incorrect, or imprecise.
2. Require at least 85% correct edges and 100% displayed edges with resolvable source chunks.
3. Run a local cross-paper question set against ordinary RAG and graph-assisted RAG; record relevant-paper recall, source-backed answer rate, and user usefulness.
4. Keep extractor/schema version with every edge and rebuild only automatic edges from changed versions.

Only after the initial types meet these gates can the project add citation, limitation, research-gap, or user-research relations.

## Acceptance criteria

1. An uploaded paper becomes `indexed` and queryable even if its graph job is pending or failed.
2. The graph worker never runs while a PDF upload job is queued or processing.
3. Every persisted edge has exactly one existing source chunk, a whitelisted type, a stable idempotent ID, and a safe provenance record.
4. Deleting a paper removes only its automatic graph edges/nodes that have no remaining source; it preserves other papers and user-confirmed information.
5. A graph relationship answer cites original chunks through the existing answer pipeline; edge text alone cannot substantiate an answer.
6. The relation explorer can show graph status and allow a user to confirm or reject a relation without reprocessing the PDF.
7. Graph extraction errors do not delete or downgrade the original paper, chunks, vectors, or regular RAG behavior.
