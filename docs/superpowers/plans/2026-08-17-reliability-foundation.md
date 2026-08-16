# Research Assistant Reliability Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make Paper Agent reproducible, evidence-aware, measurable, and safe to maintain without changing its research workflow.

**Architecture:** A pure core.evidence module validates answer citations against RetrievedChunk data before the LLM Critic selects the next route. Pure evaluation metrics reuse that report offline. Git cleanup removes generated files from the index only, never the user's local research data.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, Pytest, Docker Compose, MongoDB, Milvus.

---

## File structure

- .gitignore: local artifacts only; source and portable launch scripts remain tracked.
- .env.example, requirements.txt, requirements-dev.txt, scripts/start.ps1: reproducible local setup.
- core/evidence.py: citation parsing and deterministic evidence report.
- evaluation/metrics.py, evaluation/cases.example.json, scripts/evaluate.py: offline quality measurement.
- state/graph_state.py, agents/critic.py: evidence gate wired into the existing graph.
- tests/core/test_evidence.py, tests/evaluation/test_metrics.py, tests/test_runtime_config.py: deterministic tests.
- README.md: research workflow, accuracy limits, setup and Git conventions.

### Task 1: Make generated files untracked and ignored

**Files:**
- Modify: .gitignore
- Index only: .idea/, cache/, tmp_pdfs/

- [ ] **Step 1: Validate the exact index targets before changing them**

Run:

~~~powershell
git ls-files .idea cache tmp_pdfs
~~~

Expected: every listed file is IDE metadata, a JSON cache entry, or a downloaded PDF.

- [ ] **Step 2: Replace local-only ignore rules with this policy**

~~~gitignore
# Editors and platform files
.idea/
.vscode/
.DS_Store
Thumbs.db

# Python tooling
venv/
.venv/
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
coverage.xml
htmlcov/
.mypy_cache/
.ruff_cache/
*.log

# Credentials and local runtime data
.env
.local-data/
milvus_data/
mongodb_data/
cache/
tmp_pdfs/
uploads/
models/
evaluation/cases.local.json
evaluation/results/
~~~

- [ ] **Step 3: Verify the newly ignored contract**

~~~powershell
git check-ignore -v .idea/workspace.xml cache/example.json tmp_pdfs/example.pdf evaluation/cases.local.json
~~~

Expected: all four files match a visible rule.

- [ ] **Step 4: Remove only generated artifacts from the index and retain local files**

~~~powershell
git rm -r --cached -- .idea cache tmp_pdfs
Test-Path cache
Test-Path tmp_pdfs
~~~

Expected: git rm reports index removals; both Test-Path commands return True.

- [ ] **Step 5: Commit the hygiene boundary**

~~~powershell
git add .gitignore
git add -u -- .idea cache tmp_pdfs
git commit -m "chore: stop tracking local artifacts"
~~~

### Task 2: Make the runtime portable and test its configuration

**Files:**
- Create: .env.example, requirements.txt, requirements-dev.txt, scripts/start.ps1, tests/test_runtime_config.py
- Modify: docker-compose.yml

- [ ] **Step 1: Write failing portability tests**

~~~python
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_example_environment_has_no_secret_value():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LLM_API_KEY=" in content
    assert "sk-" not in content

def test_compose_uses_a_portable_data_root():
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${PA_DATA_ROOT:-./.local-data}" in content
    assert "E:\\milvus" not in content
~~~

- [ ] **Step 2: Run the test to observe the current failure**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/test_runtime_config.py -v
~~~

Expected: missing .env.example and fixed E:\milvus make the test fail.

- [ ] **Step 3: Add the environment, dependency and launch contracts**

Create .env.example:

~~~dotenv
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=paper_agent
MILVUS_HOST=localhost
MILVUS_PORT=19530
MINERU_URL=http://localhost:8888
USE_MCP=true
MCP_ARXIV_URL=http://localhost:8060/sse
SEMANTIC_SCHOLAR_API_KEY=
PA_DATA_ROOT=./.local-data
~~~

Create requirements.txt containing the direct imports absent from the current file: sentence-transformers, numpy, httpx, and psutil, alongside the existing LangChain, FastAPI, storage, PDF and MCP dependencies. Create requirements-dev.txt:

~~~text
-r requirements.txt
pytest
pytest-asyncio
~~~

For each etcd, minio, and milvus Compose bind mount, use this exact form:

~~~yaml
- ${PA_DATA_ROOT:-./.local-data}/etcd:/etcd
~~~

Create scripts/start.ps1:

~~~powershell
param([string]$Python = "python")
$ErrorActionPreference = "Stop"
docker compose up -d
& $Python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --log-level info
~~~

- [ ] **Step 4: Verify setup contracts**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/test_runtime_config.py -v
docker compose --env-file .env.example config --quiet
~~~

Expected: both commands exit with status 0.

- [ ] **Step 5: Commit the portable runtime**

~~~powershell
git add .env.example requirements.txt requirements-dev.txt docker-compose.yml scripts/start.ps1 tests/test_runtime_config.py
git commit -m "chore: make local setup reproducible"
~~~

### Task 3: Enforce citation evidence before semantic Critic review

**Files:**
- Create: core/evidence.py, tests/core/test_evidence.py
- Modify: state/graph_state.py, agents/critic.py

- [ ] **Step 1: Write the failing pure-function tests**

~~~python
from core.evidence import validate_answer_evidence

CHUNKS = [{"paper_title": "Attention Is All You Need", "chunk_index": 3, "score": 0.91, "metadata": {}}]

def test_accepts_a_retrieved_citation():
    report = validate_answer_evidence("结论来自 **[Attention Is All You Need]**。", CHUNKS)
    assert report["status"] == "pass"
    assert report["matched_citations"] == ["Attention Is All You Need"]

def test_rejects_a_missing_citation():
    report = validate_answer_evidence("结论来自 **[Unknown Paper]**。", CHUNKS)
    assert report["status"] == "retry"
    assert report["missing_citations"] == ["Unknown Paper"]

def test_allows_an_explicit_no_evidence_abstention():
    report = validate_answer_evidence("我无法根据现有论文证据回答。", [])
    assert report["status"] == "pass"
    assert report["reason"] == "no_evidence_abstention"
~~~

- [ ] **Step 2: Run the tests and confirm the module is missing**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/core/test_evidence.py -v
~~~

Expected: collection fails with ModuleNotFoundError for core.evidence.

- [ ] **Step 3: Implement the deterministic evidence report**

Create core/evidence.py:

~~~python
import re

_CITATION_RE = re.compile(r"\*\*\[([^\]]+)\]\*\*|(?<!\*)\[([^\]]+)\](?!\*)")
_ABSTENTION_MARKERS = ("证据不足", "无法根据现有论文证据", "未找到相关信息")

def extract_citations(answer: str) -> list[str]:
    return list(dict.fromkeys((a or b).strip() for a, b in _CITATION_RE.findall(answer) if (a or b).strip()))

def validate_answer_evidence(answer: str, chunks: list[dict]) -> dict:
    citations = extract_citations(answer)
    titles = {str(c.get("paper_title", "")).strip().casefold() for c in chunks}
    missing = [c for c in citations if c.casefold() not in titles]
    abstains = any(marker in answer for marker in _ABSTENTION_MARKERS)
    if not chunks:
        status, reason = (("pass", "no_evidence_abstention") if abstains else ("retry", "no_chunks_without_abstention"))
    elif missing:
        status, reason = "retry", "citation_not_retrieved"
    elif citations:
        status, reason = "pass", "all_citations_retrieved"
    else:
        status, reason = "retry", "answer_has_no_citations"
    return {"status": status, "reason": reason, "citations": citations, "matched_citations": [c for c in citations if c not in missing], "missing_citations": missing, "source_count": len(chunks)}
~~~

Add evidence_report: Optional[dict] to AgentState. In CriticAgent.invoke, compute the report before constructing the LLM prompt. When report status is retry and iteration is below max_iterations - 1, return evidence_report, next_agent set to retriever, and iteration plus one. Otherwise return the report alongside the existing critic result.

- [ ] **Step 4: Verify the pure gate and existing imports**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/core/test_evidence.py -v
D:\conda\envs\paper-agent\python.exe -B -c "from agents.critic import CriticAgent; print('evidence import ok')"
~~~

Expected: three passing tests and evidence import ok.

- [ ] **Step 5: Commit the evidence gate**

~~~powershell
git add core/evidence.py state/graph_state.py agents/critic.py tests/core/test_evidence.py
git commit -m "feat: validate citations against retrieval evidence"
~~~

### Task 4: Add offline research-quality metrics

**Files:**
- Create: evaluation/__init__.py, evaluation/metrics.py, evaluation/cases.example.json, scripts/evaluate.py, tests/evaluation/test_metrics.py

- [ ] **Step 1: Write failing metric tests**

~~~python
from evaluation.metrics import compute_recall_at_k, summarize_runs

def test_recall_at_k_counts_expected_papers():
    assert compute_recall_at_k(["a", "b"], ["b", "c", "a"], 2) == 0.5

def test_summary_reports_research_metrics():
    report = summarize_runs([
        {"expected_papers": ["a"], "retrieved_papers": ["a"], "evidence_status": "pass", "latency_ms": 120},
        {"expected_papers": [], "retrieved_papers": [], "evidence_status": "pass", "latency_ms": 80, "should_abstain": True},
    ])
    assert report == {"case_count": 2, "recall_at_5": 1.0, "citation_pass_rate": 1.0, "abstention_accuracy": 1.0, "latency_ms_p50": 100.0}
~~~

- [ ] **Step 2: Implement and test the metric module**

~~~python
from statistics import median

def compute_recall_at_k(expected, retrieved, k):
    return 1.0 if not expected else len(set(expected) & set(retrieved[:k])) / len(set(expected))

def summarize_runs(runs):
    total = len(runs)
    recalls = [compute_recall_at_k(r["expected_papers"], r["retrieved_papers"], 5) for r in runs]
    passed = [r.get("evidence_status") == "pass" for r in runs]
    abstain = [r for r in runs if r.get("should_abstain")]
    correct = [not r.get("retrieved_papers") and r.get("evidence_status") == "pass" for r in abstain]
    latencies = [float(r["latency_ms"]) for r in runs]
    return {"case_count": total, "recall_at_5": sum(recalls) / total if total else 0.0, "citation_pass_rate": sum(passed) / total if total else 0.0, "abstention_accuracy": sum(correct) / len(abstain) if abstain else 1.0, "latency_ms_p50": median(latencies) if latencies else 0.0}
~~~

Create cases.example.json with one normal labelled query and one should_abstain true query. scripts/evaluate.py requires --cases, rejects an absent file, runs the existing RetrieverAgent, and prints json.dumps(summarize_runs(runs), ensure_ascii=False, indent=2).

- [ ] **Step 3: Verify and commit metrics**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/evaluation/test_metrics.py -v
D:\conda\envs\paper-agent\python.exe scripts/evaluate.py --help
git add evaluation scripts/evaluate.py tests/evaluation/test_metrics.py
git commit -m "feat: add research retrieval evaluation metrics"
~~~

Expected: two tests pass and help lists --cases without connecting to MongoDB or Milvus.

### Task 5: Document, verify, and commit the foundation

**Files:**
- Create: README.md

- [ ] **Step 1: Write these README sections**

项目目标、架构、快速开始、环境变量、日常研究工作流、准确性边界、评测、测试、数据与隐私、Git 约定。The accuracy section states that unsupported answers are refused rather than inferred.

- [ ] **Step 2: Run full verification before documenting success**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest -q
docker compose --env-file .env.example config --quiet
D:\conda\envs\paper-agent\python.exe -B -c "import main; from web.app import app; print(len(app.routes))"
git check-ignore -v cache/verification.json tmp_pdfs/verification.pdf .idea/workspace.xml
~~~

Expected: all tests pass, Compose validates, the import prints a positive route count, and generated artifacts are ignored.

- [ ] **Step 3: Commit the research operations guide**

~~~powershell
git add README.md
git commit -m "docs: document reliable research workflow"
~~~

