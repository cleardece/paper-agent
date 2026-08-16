# MinerU Memory Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local MinerU start only for a parse, release its model immediately after the last parse by default, and never silently index a low-accuracy fallback result when accurate parsing is required.

**Architecture:** Add a synchronous, dependency-injected lifecycle manager between `PDFParser` and local Docker Compose. It owns health checks, reference counting, optional resource limits, and the idle stop timer, so every existing parser consumer gets identical behavior. Keep remote MinerU URLs unmanaged; expose every local-resource decision as an environment variable.

**Tech Stack:** Python 3.12, `httpx`, `subprocess`, `threading`, FastAPI, Docker Compose, pytest.

---

### Task 1: Add a testable MinerU container lifecycle manager

**Files:**
- Create: `tools/mineru_lifecycle.py`
- Create: `tests/tools/test_mineru_lifecycle.py`

- [ ] **Step 1: Write failing lifecycle tests**

```python
from tools.mineru_lifecycle import MinerUContainerManager


def test_starts_and_stops_local_parser_for_one_lease():
    commands = []
    manager = MinerUContainerManager(
        base_url="http://127.0.0.1:8888",
        idle_shutdown_seconds=0,
        command_runner=lambda args: commands.append(args),
        health_check=lambda: True,
    )

    with manager.lease():
        assert commands == [["docker", "compose", "up", "-d", "mineru-api"]]

    assert commands[-1] == ["docker", "compose", "stop", "mineru-api"]


def test_second_lease_prevents_stop_until_all_parses_finish():
    commands = []
    manager = MinerUContainerManager(
        base_url="http://127.0.0.1:8888",
        idle_shutdown_seconds=0,
        command_runner=lambda args: commands.append(args),
        health_check=lambda: True,
    )

    first = manager.lease()
    second = manager.lease()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert ["docker", "compose", "stop", "mineru-api"] not in commands
    second.__exit__(None, None, None)

    assert commands[-1] == ["docker", "compose", "stop", "mineru-api"]


def test_configured_limits_are_applied_without_a_global_default():
    commands = []
    manager = MinerUContainerManager(
        base_url="http://127.0.0.1:8888",
        memory_limit="8g",
        cpu_limit="2.0",
        command_runner=lambda args: commands.append(args),
        health_check=lambda: True,
    )

    with manager.lease():
        pass

    assert ["docker", "update", "--memory", "8g", "--cpus", "2.0", "mineru-api"] in commands
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/tools/test_mineru_lifecycle.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.mineru_lifecycle'`.

- [ ] **Step 3: Implement the manager with one command boundary**

```python
# tools/mineru_lifecycle.py
from __future__ import annotations

from contextlib import contextmanager
import subprocess
import threading
import time
from collections.abc import Callable, Iterator


class MinerUContainerError(RuntimeError):
    pass


class MinerUContainerManager:
    def __init__(self, base_url: str, *, idle_shutdown_seconds: int = 0,
                 start_timeout_seconds: int = 90, memory_limit: str = "",
                 cpu_limit: str = "", command_runner: Callable[[list[str]], None] | None = None,
                 health_check: Callable[[], bool] | None = None):
        self.base_url = base_url.rstrip("/")
        self.idle_shutdown_seconds = idle_shutdown_seconds
        self.start_timeout_seconds = start_timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._command_runner = command_runner or self._run_command
        self._health_check = health_check or self._is_healthy
        self._lock = threading.Lock()
        self._active_parses = 0
        self._stop_timer: threading.Timer | None = None

    @contextmanager
    def lease(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def acquire(self) -> None:
        with self._lock:
            if self._stop_timer:
                self._stop_timer.cancel()
                self._stop_timer = None
            if self._active_parses == 0:
                self._ensure_running()
            self._active_parses += 1

    def release(self) -> None:
        with self._lock:
            self._active_parses -= 1
            if self._active_parses == 0:
                if self.idle_shutdown_seconds == 0:
                    self._stop()
                else:
                    self._stop_timer = threading.Timer(self.idle_shutdown_seconds, self._stop_if_idle)
                    self._stop_timer.daemon = True
                    self._stop_timer.start()
```

Implement `_ensure_running()` with `docker compose up -d mineru-api`, a polling loop bounded by `start_timeout_seconds`, and optional `docker update` only when at least one limit is configured. Implement `_stop_if_idle()` under `_lock`, `_stop()` with `docker compose stop mineru-api`, `_run_command()` with `subprocess.run(..., check=True, timeout=60, cwd=repository_root)`, and `_is_healthy()` with `httpx.get(f"{self.base_url}/health", timeout=5)`. Convert command, timeout, and health failures to `MinerUContainerError` containing the original diagnostic text.

- [ ] **Step 4: Run lifecycle tests**

Run: `python -m pytest tests/tools/test_mineru_lifecycle.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit the isolated lifecycle component**

```bash
git add tools/mineru_lifecycle.py tests/tools/test_mineru_lifecycle.py
git commit -m "feat: manage MinerU container lifecycle"
```

### Task 2: Make parser accuracy policy explicit and use the lifecycle manager

**Files:**
- Modify: `config.py:91-94`
- Modify: `core/deps.py:8, 45-59`
- Modify: `tools/pdf_parser.py:15-56`
- Modify: `tests/tools/test_pdf_parser_mineru.py`
- Modify: `tests/test_runtime_config.py`

- [ ] **Step 1: Add failing strict-accuracy and remote-service tests**

```python
import pytest
from tools.pdf_parser import MinerUParseError, PDFParser


def test_accurate_mode_does_not_silently_fallback_when_mineru_fails(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parser = PDFParser(mineru_url="http://mineru.test", require_accurate_parse=True)
    monkeypatch.setattr(parser, "_parse_with_mineru", lambda _: (_ for _ in ()).throw(RuntimeError("OOM")))

    with pytest.raises(MinerUParseError, match="OOM"):
        parser.parse(str(pdf_path))


def test_non_strict_mode_keeps_explicit_pdfplumber_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parser = PDFParser(mineru_url="http://mineru.test", require_accurate_parse=False)
    monkeypatch.setattr(parser, "_parse_with_mineru", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(parser, "_parse_with_pdfplumber", lambda _: {"source": "pdfplumber"})

    assert parser.parse(str(pdf_path))["source"] == "pdfplumber"
```

Add a runtime-config assertion that `.env.example` documents `MINERU_IDLE_SHUTDOWN_SECONDS=0`, `MINERU_REQUIRE_ACCURATE_PARSE=true`, and the optional CPU/memory variables without values.

- [ ] **Step 2: Run the parser and config tests to verify they fail**

Run: `python -m pytest tests/tools/test_pdf_parser_mineru.py tests/test_runtime_config.py -v`

Expected: FAIL because `MinerUParseError` and the new constructor arguments do not exist.

- [ ] **Step 3: Add configuration and parser integration**

```python
# config.py
MINERU_IDLE_SHUTDOWN_SECONDS = int(os.getenv("MINERU_IDLE_SHUTDOWN_SECONDS", "0"))
MINERU_START_TIMEOUT_SECONDS = int(os.getenv("MINERU_START_TIMEOUT_SECONDS", "90"))
MINERU_MEMORY_LIMIT = os.getenv("MINERU_MEMORY_LIMIT", "")
MINERU_CPU_LIMIT = os.getenv("MINERU_CPU_LIMIT", "")
MINERU_REQUIRE_ACCURATE_PARSE = os.getenv("MINERU_REQUIRE_ACCURATE_PARSE", "true").lower() == "true"
```

In `core/deps.py`, construct `MinerUContainerManager` only when `MINERU_URL` points to `localhost`, `127.0.0.1`, or `::1`; inject it and `MINERU_REQUIRE_ACCURATE_PARSE` into `PDFParser`. In `PDFParser.parse()`, wrap only `_parse_with_mineru()` in `with self.mineru_manager.lease()` when a manager exists. On an exception, raise `MinerUParseError` when strict mode is enabled; otherwise log the explicit downgrade and call `_parse_with_pdfplumber()`.

- [ ] **Step 4: Run parser and config tests**

Run: `python -m pytest tests/tools/test_pdf_parser_mineru.py tests/test_runtime_config.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit parser policy changes**

```bash
git add config.py core/deps.py tools/pdf_parser.py tests/tools/test_pdf_parser_mineru.py tests/test_runtime_config.py
git commit -m "feat: enforce accurate MinerU parsing policy"
```

### Task 3: Preserve parse failure state and make the default Compose startup memory-safe

**Files:**
- Modify: `web/app.py:527-687`
- Modify: `web/static/papers.html:57-65, 138-144`
- Modify: `docker-compose.yml:67-86`
- Modify: `scripts/start.ps1:7`
- Create: `tests/web/test_upload_parse_failure.py`

- [ ] **Step 1: Write a failing upload-failure test**

```python
import asyncio

from tools.pdf_parser import MinerUParseError
from web.app import _process_upload


def test_upload_marks_paper_as_parse_failed_when_accurate_parse_fails(tmp_path):
    updates = []
    container = type("Container", (), {})()
    container.pdf_parser = type("Parser", (), {"parse": lambda *_: (_ for _ in ()).throw(MinerUParseError("MinerU 内存不足"))})()
    container.mongodb = type("Mongo", (), {"upsert_paper": lambda _, paper: updates.append(paper)})()

    asyncio.run(_process_upload(container, str(tmp_path / "paper.pdf"), "paper.pdf", "local_test"))

    assert updates == [{
        "arxiv_id": "local_test", "title": "paper.pdf", "status": "parse_failed",
        "parse_error": "MinerU 内存不足", "parser_source": "mineru",
    }]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/web/test_upload_parse_failure.py -v`

Expected: FAIL because `_process_upload()` only logs the exception and does not persist a failure state.

- [ ] **Step 3: Implement failure persistence and parser profile startup**

In `_process_upload()`, catch `MinerUParseError` before the generic exception and call `container.mongodb.upsert_paper()` with exactly the fields shown in the test. In `upload_paper()`, permit a new upload for an existing `parse_failed` record by deleting that record and its chunks before scheduling the replacement task; keep the existing 409 behavior for every other status.

Add `profiles: ["parser"]` to `mineru-api` and change its healthcheck to `curl -sf http://localhost:8000/health`. This prevents `docker compose up -d` and `scripts/start.ps1` from starting the high-memory parser at application boot; `docker compose up -d mineru-api` used by the lifecycle manager still starts it on demand. Add `.paper-status.parse_failed` styling and render `parse_error` as a title attribute on the status badge in `papers.html`.

- [ ] **Step 4: Run the upload test and Compose validation**

Run: `python -m pytest tests/web/test_upload_parse_failure.py -v`

Expected: 1 passed.

Run: `docker compose --env-file .env.example config --quiet`

Expected: exit code 0.

- [ ] **Step 5: Commit the visible failure path and lazy Compose profile**

```bash
git add web/app.py web/static/papers.html docker-compose.yml scripts/start.ps1 tests/web/test_upload_parse_failure.py
git commit -m "feat: release MinerU after local parse"
```

### Task 4: Document resource profiles and verify the integrated behavior

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_runtime_config.py`

- [ ] **Step 1: Add documentation assertions**

```python
def test_readme_documents_parser_resource_profiles():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "MINERU_IDLE_SHUTDOWN_SECONDS=0" in content
    assert "MINERU_CPU_LIMIT" in content
    assert "MINERU_MEMORY_LIMIT" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_runtime_config.py::test_readme_documents_parser_resource_profiles -v`

Expected: FAIL because the resource-policy documentation is absent.

- [ ] **Step 3: Document safe defaults and opt-in tuning**

Add to `.env.example`:

```dotenv
# Default: stop MinerU immediately after each parse; set e.g. 300 only for batch uploads.
MINERU_IDLE_SHUTDOWN_SECONDS=0
MINERU_REQUIRE_ACCURATE_PARSE=true
# Optional host-specific limits. Leave blank to avoid forcing one machine profile on all users.
MINERU_MEMORY_LIMIT=
MINERU_CPU_LIMIT=
```

In `README.md`, document the default immediate-release lifecycle, the cold-start tradeoff, the optional `MINERU_MEMORY_LIMIT=8g` and `MINERU_CPU_LIMIT=2.0` examples, and that strict mode marks MinerU errors as `parse_failed` rather than silently indexing pdfplumber output.

- [ ] **Step 4: Run the focused and full verification suite**

Run: `python -m pytest tests/test_runtime_config.py -v`

Expected: all runtime config tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

Run: `python -c "from core.deps import get_container; print('container import ok')"`

Expected: `container import ok` after the existing service initialization logs.

- [ ] **Step 5: Commit docs and final verification changes**

```bash
git add .env.example README.md tests/test_runtime_config.py
git commit -m "docs: explain portable MinerU resource settings"
```
