# Observable Research UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Preserve the current chat and paper-library workflow while making evidence, quality decisions and Agent execution understandable in a polished light/dark research interface.

**Architecture:** The FastAPI completion event carries the existing timeline plus evidence_report from the reliability plan. The framework-free frontend renders safe DOM nodes for evidence and timing. CSS custom properties provide both themes, and the paper-library page uses the same local preference.

**Tech Stack:** FastAPI streaming responses, WebSocket status events, vanilla JavaScript, CSS custom properties, Pytest.

---

## File structure

- web/app.py: duration and evidence fields on persisted messages and final SSE events.
- tests/web/test_chat_presentation.py: response contract without MongoDB or Milvus.
- web/static/index.html: theme control and stable chat header.
- web/static/app.js: theme persistence, evidence rendering, stream-error recovery.
- web/static/style.css: responsive light/dark design system and component states.
- web/static/papers.html: same preference and visual language without changing paper actions.
- tests/web/test_static_ui.py: static accessibility and integration contracts.

### Task 1: Add duration and evidence to the web response contract

**Files:**
- Modify: web/app.py
- Create: tests/web/test_chat_presentation.py

- [ ] **Step 1: Write failing contract tests**

~~~python
from web.app import ChatMessage, Session, serialize_session, timeline_snapshot

def test_timeline_preserves_agent_duration():
    timeline = timeline_snapshot([{"agent": "retriever", "status": "completed", "duration_ms": 83.4}])
    step = next(item for item in timeline if item["agent"] == "retriever")
    assert step["duration_ms"] == 83.4

def test_session_serialization_includes_evidence_report():
    session = Session(id="s1", title="test")
    session.messages.append(ChatMessage(role="assistant", content="answer", evidence_report={"status": "pass"}))
    payload = serialize_session(session, include_messages=True)
    assert payload["messages"][0]["evidence_report"]["status"] == "pass"
~~~

- [ ] **Step 2: Run the test to capture the absent contract fields**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_chat_presentation.py -v
~~~

Expected: duration_ms and ChatMessage.evidence_report are missing.

- [ ] **Step 3: Extend the contract without breaking saved sessions**

In web/app.py, extend ChatMessage exactly as follows:

~~~python
@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    evidence_report: dict[str, Any] | None = None
~~~

Initialize each timeline_snapshot step with duration_ms: None. In wrap_agent, record started_at with time.perf_counter immediately before run_sync_or_async and include rounded elapsed milliseconds in completed and error status payloads. Include evidence_report with a default of None while loading, saving and serializing messages. In the chat endpoint, obtain final_result evidence_report, save it in the assistant message, and include it in the done SSE payload.

- [ ] **Step 4: Verify contract compatibility**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_chat_presentation.py -v
D:\conda\envs\paper-agent\python.exe -B -c "from web.app import app; print('routes', len(app.routes))"
~~~

Expected: two passing tests and a positive route count.

- [ ] **Step 5: Commit the observable response contract**

~~~powershell
git add web/app.py tests/web/test_chat_presentation.py
git commit -m "feat: expose evidence and agent timings in chat"
~~~

### Task 2: Add accessible theme controls without changing the user flow

**Files:**
- Modify: web/static/index.html, web/static/papers.html
- Create: tests/web/test_static_ui.py

- [ ] **Step 1: Write failing static interface tests**

~~~python
from pathlib import Path

ROOT = Path(__file__).parents[2]

def test_chat_has_accessible_theme_toggle():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    assert 'id="themeToggle"' in html
    assert 'aria-label="切换深浅主题"' in html

def test_paper_library_uses_shared_theme_contract():
    html = (ROOT / "web/static/papers.html").read_text(encoding="utf-8")
    assert 'data-theme="dark"' in html
    assert 'id="themeToggle"' in html
~~~

- [ ] **Step 2: Run the tests and confirm the controls do not exist**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -v
~~~

Expected: both assertions fail.

- [ ] **Step 3: Add semantic header controls**

Set data-theme="dark" on each html element. Replace the index topbar content with:

~~~html
<header class="topbar">
  <div class="workspace-title">
    <span class="eyebrow">研究工作台</span>
    <strong id="chatTitle">新对话</strong>
    <span id="connectionState" class="connection-state">准备就绪</span>
  </div>
  <button id="themeToggle" class="icon-button" type="button" aria-label="切换深浅主题" title="切换深浅主题">◐</button>
</header>
~~~

Add the same button next to the existing return link in papers.html. Preserve every existing upload, question, delete, session and API action.

- [ ] **Step 4: Verify and commit controls**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/web/test_static_ui.py -v
git add web/static/index.html web/static/papers.html tests/web/test_static_ui.py
git commit -m "feat: add shared interface theme controls"
~~~

Expected: two passing tests.

### Task 3: Implement an accessible research design system

**Files:**
- Modify: web/static/style.css

- [ ] **Step 1: Define the light and dark semantic tokens**

~~~css
:root,
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #f7f8fc;
  --panel: #ffffff;
  --surface-raised: #ffffff;
  --surface-subtle: #f0f3f9;
  --bubble: #ffffff;
  --bubble-user: #2563eb;
  --text: #172033;
  --muted: #667085;
  --line: #dce2ee;
  --border: #e3e8f2;
  --accent: #6d5dfc;
  --accent-strong: #5545e8;
  --success: #16a36a;
  --warning: #ba7a00;
  --danger: #dc3c52;
  --shadow: 0 14px 36px rgb(23 32 51 / 8%);
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #10131a;
  --panel: #171b25;
  --surface-raised: #1d2330;
  --surface-subtle: #242b3a;
  --bubble: #1d2432;
  --bubble-user: #5b52e8;
  --text: #edf1fb;
  --muted: #aab4c5;
  --line: #3a4354;
  --border: #30394a;
  --accent: #9b92ff;
  --accent-strong: #b0a9ff;
  --success: #45ca90;
  --warning: #e8b34a;
  --danger: #ff8291;
  --shadow: 0 18px 42px rgb(0 0 0 / 26%);
}
~~~

- [ ] **Step 2: Apply only semantic colours and add required states**

Replace literal component colours with the token variables. Add styles for icon-button, workspace-title, eyebrow, connection-state, evidence-panel, evidence-status.pass, evidence-status.retry, duration, empty-state and button:focus-visible. Retain the current 760px breakpoint and use the existing class names for sidebar, composer, message and timeline.

- [ ] **Step 3: Check visual accessibility in both themes**

Run the server and check / and /papers: theme persists after refresh and navigation; user and assistant text has strong contrast; tab focus is visible; upload errors, empty library and timeline labels remain legible.

- [ ] **Step 4: Commit the design system**

~~~powershell
git add web/static/style.css
git commit -m "feat: add accessible research workspace themes"
~~~

### Task 4: Render evidence and execution information in chat

**Files:**
- Modify: web/static/app.js

- [ ] **Step 1: Add the theme bootstrap**

Immediately after the DOM queries, set the root theme from localStorage key paperAgentTheme, defaulting to dark. Add a click listener on themeToggle that flips dark/light and persists the new value.

~~~javascript
const themeToggle = document.querySelector("#themeToggle");
const savedTheme = localStorage.getItem("paperAgentTheme");
document.documentElement.dataset.theme = savedTheme || "dark";
themeToggle?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("paperAgentTheme", next);
});
~~~

- [ ] **Step 2: Extend timeline rendering**

In renderTimeline, calculate duration as an empty string when duration_ms is null, otherwise a Chinese millisecond label. Append it to the existing agent-status alongside retry count; do not remove current input/output details.

- [ ] **Step 3: Add a safe evidence panel**

Add renderEvidence(report). It creates a details element with class evidence-panel, assigns pass only when report.status equals pass, and inserts escaped matched_citations, missing_citations, reason and source_count. Change addMessage signature to accept evidenceReport. Render its panel before the answer bubble when present. While loading session history, pass message.evidence_report; while processing done SSE, render data.evidence_report.

- [ ] **Step 4: Recover from stream failure**

Wrap the chat fetch and reader loop in try/catch/finally. Catch sets the current assistant bubble to: 网络或服务异常，未生成可引用的研究结论。请检查服务状态后重试。 Finally always sets sendButton.disabled to false. Preserve the current websocket, upload and pending-paper-question flow.

- [ ] **Step 5: Manually verify the three research cases**

Verify one answer with matched evidence, one answer with insufficient evidence, and one request during a stopped server. Confirm distinct panel status, duration labels, persistent theme and a re-enabled send button.

- [ ] **Step 6: Commit the evidence-centred chat UI**

~~~powershell
git add web/static/app.js
git commit -m "feat: show answer evidence and execution timings"
~~~

### Task 5: Finish paper-library consistency and verify the UI

**Files:**
- Modify: web/static/papers.html

- [ ] **Step 1: Bootstrap the shared preference**

Before the existing paper-library script, set document.documentElement.dataset.theme from paperAgentTheme and add the same toggle listener. Replace inline literal page colours with surface-raised, surface-subtle, border, text, muted, accent and danger tokens. Keep loadPapers, askAbout and deletePaper names and endpoint URLs unchanged.

- [ ] **Step 2: Run final verification**

~~~powershell
D:\conda\envs\paper-agent\python.exe -m pytest tests/web -v
D:\conda\envs\paper-agent\python.exe -B -c "from web.app import app; print('routes', len(app.routes))"
~~~

Then manually verify both pages in both themes, paper deletion confirmation, and the 提问 navigation.

- [ ] **Step 3: Commit the paper library refinement**

~~~powershell
git add web/static/papers.html
git commit -m "feat: align paper library with research workspace theme"
~~~

