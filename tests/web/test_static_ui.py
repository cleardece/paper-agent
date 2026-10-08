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


def test_research_graph_page_exposes_evidence_review_and_empty_contract():
    html = (ROOT / "web/static/graph.html").read_text(encoding="utf-8")
    assert "/api/research-graph" in html
    assert "原文证据" in html
    assert "confirmed" in html and "rejected" in html
    assert "未发现可证据化关系" in html
    assert "/api/research-graph/jobs/retry" in html


def test_research_graph_polling_stops_when_idle_or_hidden():
    html = (ROOT / "web/static/graph.html").read_text(encoding="utf-8")
    assert "scheduleDashboardRefresh" in html
    assert 'document.addEventListener("visibilitychange"' in html
    assert "setInterval(" not in html


def test_upload_supports_multiple_files_and_batch_progress_panel():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    script = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
    assert 'id="fileInput"' in html and "multiple" in html
    assert 'id="uploadBatchPanel"' in html
    assert 'fetch(`/api/upload-batches/${currentUploadBatchId}`)' in script
    assert 'localStorage.setItem("paperAgentLastUploadBatch"' in script


def test_upload_ui_has_draft_submit_and_recent_history_contract():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    script = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
    assert 'id="uploadDraftPanel"' in html
    assert 'id="startUploadBtn"' in html
    assert 'id="recentUploadBatches"' in html
    assert "const draftFiles = new Map()" in script
    assert 'fetch("/api/upload-batches?days=7&limit=20")' in script
    assert 'fileInput.addEventListener("change", addFilesToDraft)' in script
