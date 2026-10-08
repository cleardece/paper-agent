# Paper Agent

[中文](README.md) · [Architecture](docs/architecture.md) · [Operations](docs/operations.md) · [Changelog](CHANGELOG.md)

A personal research assistant for paper discovery, PDF ingestion, evidence-backed question answering, multi-turn paper context, and reviewable knowledge graphs.

## Capabilities

- Search academic papers or ingest local PDFs through the MinerU API.
- Retrieve indexed passages with BM25, vector search, fusion, and reranking.
- Route requests through LangGraph agents while preserving stable paper identities.
- Rewrite contextual follow-up questions without treating conversation history as evidence.
- Build Evidence Graph V4 with canonical entities, claims, facts, and provenance in MongoDB and Milvus.
- Run graph extraction with an independently configured model and persistent task checkpoints.

## Getting started

Requirements: Python 3.10+, Docker, an OpenAI-compatible chat endpoint, a MinerU API token, and a suitable PyTorch installation for local embeddings. The sample graph configuration additionally requires Ollama.

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
# Fill in LLM_API_KEY and MINERU_OFFICIAL_TOKEN; review the endpoint settings.
python -m pip install -r requirements.txt
ollama pull qwen3:8b
ollama create qwen3-graph:8b -f Modelfile
docker compose --env-file .env up -d
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Keep Ollama running. Open `http://localhost:8000`; the library is at `/papers` and the graph at `/graph`. Configuration details and operational guidance are in the [Chinese README](README.md) and [operations guide](docs/operations.md).

`LLM_*` configures chat; `GRAPH_LLM_*` configures graph extraction. The sample graph endpoint is local Ollama at `http://127.0.0.1:11434/v1`. `GRAPH_LLM_TPM_LIMIT=0` disables proactive token throttling, not graph extraction.

## Development and evidence boundaries

Install `requirements-dev.txt` for the tests. Some tests require live MongoDB, Milvus, or model dependencies. See the [test plan](docs/test_plan.md) before choosing a test scope.

Answers should cite evidence retrieved for the current turn. Citation ownership checks do not establish that every claim is entailed by the source: inspect original passages before using conclusions in academic writing. A graph checkpoint or queued retry is not a completed extraction.

Keep API keys, papers, generated outputs, model weights, and database data out of Git. This repository is maintained as a personal research tool; it does not promise a production SLA. Historical design documents live under `docs/superpowers/`. The bundled `arxiv-mcp/` component carries its own license.
