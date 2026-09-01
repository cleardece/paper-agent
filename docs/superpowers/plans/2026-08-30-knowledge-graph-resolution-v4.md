# Knowledge Graph Resolution V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing evidence graph to canonical Entity/Claim/Fact resolution with complete provenance, incremental updates, and ambiguity-only batch LLM calls while preserving existing APIs and RAG behavior.

**Architecture:** Keep job lifecycle and compatibility queries in `ResearchGraphRepository`; move fixed schema, normalization, candidate retrieval/scoring, and batch ambiguity resolution into a focused `knowledge_graph` package. Persist canonical objects in MongoDB and use isolated Milvus collections for Top-K candidate retrieval; project canonical claims back into the existing edge collection.

**Tech Stack:** Python 3.13, MongoDB/PyMongo, Milvus/PyMilvus, BGE-M3 embeddings, existing LangChain LLM client, pytest.

---

### Task 1: Fixed schema and raw/canonical claim contract

**Files:**
- Create: `knowledge_graph/schema/entity_types.py`
- Create: `knowledge_graph/schema/predicates.py`
- Create: `knowledge_graph/models.py`
- Modify: `agents/research_graph_extractor.py`
- Test: `tests/knowledge_graph/test_schema.py`
- Test: `tests/agents/test_research_graph_extractor.py`

- [ ] Add immutable entity type, predicate, stance, and legacy relation mappings.
- [ ] Make extraction return raw subject/predicate/object/evidence records without accepting free canonical predicates.
- [ ] Make verification normalize entity types, predicate, qualifiers, stance, confidence, and validity in the existing second LLM call.
- [ ] Add tests proving unknown predicates become `UNKNOWN` and cannot bypass schema validation.

### Task 2: Make verification partial-output tolerant

**Files:**
- Modify: `agents/research_graph_extractor.py`
- Test: `tests/agents/test_research_graph_extractor.py`

- [ ] Add the two-candidate/one-decision regression test and verify it fails with the current `ValueError`.
- [ ] Reconcile decisions by candidate index; preserve valid decisions and synthesize `uncertain` decisions for missing or invalid entries.
- [ ] Record missing, invalid, duplicate, and returned decision counts in diagnostics.
- [ ] Verify malformed top-level JSON still fails so the existing retry path remains meaningful.

### Task 3: Entity normalization and deterministic Fast Path

**Files:**
- Create: `knowledge_graph/entity_resolution/normalizer.py`
- Create: `knowledge_graph/entity_resolution/scorer.py`
- Create: `knowledge_graph/entity_resolution/resolver.py`
- Test: `tests/knowledge_graph/test_entity_resolution.py`

- [ ] Implement Unicode/case/hyphen/whitespace/plural normalization and acronym generation.
- [ ] Implement exact canonical and exact alias lookup before embeddings.
- [ ] Implement composite candidate scoring with type as a hard constraint and name/acronym/vector/context signals.
- [ ] Verify PINN variants resolve to one entity and new aliases make future resolution LLM-free.

### Task 4: Isolated KG vector candidate index

**Files:**
- Create: `knowledge_graph/vector_index.py`
- Modify: `core/deps.py`
- Test: `tests/knowledge_graph/test_vector_index.py`

- [ ] Create `kg_entity_embeddings` and `kg_fact_embeddings` without changing existing paper/chunk collections.
- [ ] Implement idempotent entity/fact upsert, delete, and type-filtered Top-K search.
- [ ] Inject the index into the graph worker while keeping repository construction compatible with Mongo-only tests.
- [ ] Verify KG index operations never target `paper_chunks` or `paper_embeddings`.

### Task 5: Batched Entity ambiguity resolution

**Files:**
- Create: `knowledge_graph/entity_resolution/llm_resolver.py`
- Modify: `tools/research_graph_process.py`
- Modify: `tools/research_graph_worker.py`
- Test: `tests/knowledge_graph/test_entity_resolution.py`
- Test: `tests/tools/test_research_graph_worker.py`

- [ ] Collect all ambiguous mentions from one paper into one subprocess payload.
- [ ] Accept only `merge` or `new` decisions referencing supplied candidate ids.
- [ ] Conservatively create an unresolved entity when a batch decision is missing or invalid.
- [ ] Assert exact, alias, high-score, and clearly-new paths make zero Entity Resolution LLM calls.

### Task 6: Claim, Fact, stance, and provenance repositories

**Files:**
- Modify: `storage/research_graph.py`
- Create: `knowledge_graph/fact_resolution/signature.py`
- Create: `knowledge_graph/fact_resolution/scorer.py`
- Create: `knowledge_graph/fact_resolution/resolver.py`
- Test: `tests/storage/test_research_graph_canonical.py`
- Test: `tests/knowledge_graph/test_fact_resolution.py`

- [ ] Add indexed MongoDB collections for entities, claims, facts, aliases, and resolution cache.
- [ ] Build stable Fact signatures only after both endpoints have canonical entity ids.
- [ ] Preserve qualifiers on Claim and aggregate support/contradict counts on Fact.
- [ ] Persist full paper/chunk/section/page/evidence/context/hash and all processing versions.
- [ ] Incrementally recompute only facts touched by the paper being written.

### Task 7: Batched Fact ambiguity resolution

**Files:**
- Create: `knowledge_graph/fact_resolution/llm_resolver.py`
- Modify: `tools/research_graph_process.py`
- Modify: `tools/research_graph_worker.py`
- Test: `tests/knowledge_graph/test_fact_resolution.py`
- Test: `tests/tools/test_research_graph_worker.py`

- [ ] Resolve exact signatures without embeddings or LLM.
- [ ] Retrieve only Top-K semantic candidates and apply structural scoring before any LLM call.
- [ ] Send all ambiguous facts for a paper in one subprocess payload.
- [ ] Conservatively create an unresolved Fact when a decision is missing or invalid.

### Task 8: Atomic graph write and compatibility projection

**Files:**
- Create: `knowledge_graph/pipeline.py`
- Modify: `storage/research_graph.py`
- Modify: `tools/research_graph_worker.py`
- Test: `tests/storage/test_research_graph_canonical.py`
- Test: `tests/tools/test_research_graph_worker.py`

- [ ] Run verification, Entity Resolution, Fact Resolution, provenance validation, and graph write in order.
- [ ] Replace only the current paper's system claims/edges after a complete successful run.
- [ ] Project canonical claims to current edge fields and preserve confirmed/rejected review state.
- [ ] Keep `search`, `paper_links`, and `find_related_paper_ids` signatures and usable-status behavior unchanged.

### Task 9: KG evaluation and diagnostics

**Files:**
- Create: `knowledge_graph/evaluation.py`
- Modify: `storage/research_graph.py`
- Test: `tests/knowledge_graph/test_evaluation.py`

- [ ] Compute Entity/Fact duplicate rate, incorrect merge rate from reviewed cases, resolution route counts, LLM batch counts, and provenance completeness.
- [ ] Add counts to graph job diagnostics and status summary without changing existing response fields.
- [ ] Verify every usable Fact has at least one Claim with complete provenance.

### Task 10: Verification

**Files:**
- Modify only files listed above if verification exposes a task-scoped defect.

- [ ] Run focused KG and worker tests with the bundled Python plus the repository site-packages.
- [ ] Run the complete pytest suite.
- [ ] Run `python -m compileall knowledge_graph agents storage tools core`.
- [ ] Run `git diff --check` and inspect `git diff --stat` to confirm no RAG, parser, Agent workflow, API route, or UI files changed.
