const sessionsEl = document.querySelector("#sessions");
const messagesEl = document.querySelector("#messages");
const inputEl = document.querySelector("#messageInput");
const composerEl = document.querySelector("#composer");
const sendButton = document.querySelector("#sendButton");
const newChatButton = document.querySelector("#newChat");
const chatTitleEl = document.querySelector("#chatTitle");
const connectionEl = document.querySelector("#connectionState");
const themeToggle = document.querySelector("#themeToggle");

const savedTheme = localStorage.getItem("paperAgentTheme");
document.documentElement.dataset.theme = savedTheme || "dark";
themeToggle?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("paperAgentTheme", next);
});

let currentSessionId = null;
let socket = null;
let socketToken = 0;
let runningTimeline = freshTimeline();
let currentTimelineEl = null;

function freshTimeline() {
  return ["supervisor", "fetcher", "retriever", "analyzer", "critic", "presenter"].map((agent) => ({
    agent,
    status: "waiting",
    detail: "",
    input_summary: "",
    output_summary: "",
    retry_count: 0,
    duration_ms: null,
  }));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>");
}

function statusText(status) {
  return {
    waiting: "等待中",
    running: "执行中",
    completed: "完成",
    skipped: "跳过",
  }[status] || status;
}

function renderTimeline(timeline) {
  const wrapper = document.createElement("div");
  wrapper.className = "timeline";
  wrapper.innerHTML = timeline.map((step) => {
    const retry = step.agent === "critic" && step.retry_count ? ` · 重试 ${step.retry_count} 次` : "";
    const duration = Number.isFinite(step.duration_ms) ? ` · ${step.duration_ms} ms` : "";
    const input = escapeHtml(step.input_summary || "暂无输入摘要");
    const output = escapeHtml(step.output_summary || "暂无输出摘要");
    return `
      <div class="step ${step.status}">
        <span class="dot"></span>
        <div>
          <div class="step-head">
            <span class="agent-name">${step.agent}</span>
            <span class="agent-status">${statusText(step.status)}${retry}${duration}</span>
          </div>
          <div class="agent-detail">${escapeHtml(step.detail || "")}</div>
          <details>
            <summary>查看输入/输出摘要</summary>
            <div class="summary-block">输入：${input}</div>
            <div class="summary-block">输出：${output}</div>
          </details>
        </div>
      </div>
    `;
  }).join("");
  return wrapper;
}

function renderEvidence(report) {
  const panel = document.createElement("details");
  panel.className = "evidence-panel";
  const state = report?.status === "pass" ? "pass" : "retry";
  const title = state === "pass" ? "证据校验通过" : "证据需要复核";
  const matched = (report?.matched_citations || []).map(escapeHtml).join("、") || "无";
  const missing = (report?.missing_citations || []).map(escapeHtml).join("、") || "无";
  const reason = escapeHtml(report?.reason || "unknown");
  const sourceCount = Number(report?.source_count || 0);
  panel.innerHTML = `
    <summary><span class="evidence-status ${state}">${title}</span> · ${reason}</summary>
    <div class="evidence-grid">
      <div><strong>已匹配引用</strong><p>${matched}</p></div>
      <div><strong>待复核引用</strong><p>${missing}</p></div>
      <div><strong>检索片段</strong><p>${sourceCount} 个</p></div>
    </div>
  `;
  return panel;
}

function updateTimeline(event) {
  const index = runningTimeline.findIndex((step) => step.agent === event.agent);
  if (index < 0) return;
  runningTimeline[index] = { ...runningTimeline[index], ...event };
  if (event.agent === "fetcher" && event.status === "running") {
    for (const agent of ["retriever", "analyzer", "critic", "presenter"]) {
      const i = runningTimeline.findIndex((step) => step.agent === agent);
      runningTimeline[i].status = "skipped";
    }
  }
  if (event.agent === "retriever" && event.status === "running") {
    const i = runningTimeline.findIndex((step) => step.agent === "fetcher");
    runningTimeline[i].status = "skipped";
  }
  if (currentTimelineEl) {
    const next = renderTimeline(runningTimeline);
    currentTimelineEl.replaceWith(next);
    currentTimelineEl = next;
  }
}

function addMessage(role, content, timeline = null, evidenceReport = null) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (role === "assistant" && timeline) {
    row.appendChild(renderTimeline(timeline));
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${renderMarkdown(content || "")}</p>`;
  if (role === "assistant" && evidenceReport) {
    row.appendChild(renderEvidence(evidenceReport));
  }
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { row, bubble };
}

function connectSocket(sessionId) {
  const token = ++socketToken;
  if (socket) socket.close();
  connectionEl.textContent = "连接中";
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws/${sessionId}`);
  socket.onopen = () => {
    if (token === socketToken) connectionEl.textContent = "实时状态已连接";
  };
  socket.onclose = () => {
    if (token === socketToken) connectionEl.textContent = currentSessionId ? "实时状态未连接" : "准备就绪";
  };
  socket.onmessage = (message) => {
    if (token !== socketToken) return;
    const event = JSON.parse(message.data);
    if (event.type === "agent_status") updateTimeline(event);
  };
}

async function loadSessions() {
  const res = await fetch("/api/sessions");
  const sessions = await res.json();
  sessionsEl.innerHTML = "";
  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = `session-item-wrapper ${session.id === currentSessionId ? "active" : ""}`;

    const button = document.createElement("button");
    button.className = "session-item";
    button.textContent = session.title;
    button.onclick = () => loadSession(session.id);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "session-delete";
    deleteBtn.textContent = "×";
    deleteBtn.title = "删除对话";
    deleteBtn.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("确定删除这个对话？")) return;
      await fetch(`/api/sessions/${session.id}`, { method: "DELETE" });
      if (session.id === currentSessionId) {
        currentSessionId = null;
        chatTitleEl.textContent = "新对话";
        messagesEl.innerHTML = "";
      }
      await loadSessions();
    };

    item.appendChild(button);
    item.appendChild(deleteBtn);
    sessionsEl.appendChild(item);
  });
}

async function loadSession(sessionId) {
  const res = await fetch(`/api/sessions/${sessionId}`);
  if (!res.ok) return;
  const session = await res.json();
  currentSessionId = session.id;
  chatTitleEl.textContent = session.title;
  messagesEl.innerHTML = "";
  session.messages.forEach((message) => addMessage(
    message.role,
    message.content,
    message.timeline,
    message.evidence_report,
  ));
  connectSocket(session.id);
  await loadSessions();
}

async function sendMessage(text, targetPaperId = null) {
  sendButton.disabled = true;
  if (!currentSessionId) {
    currentSessionId = crypto.randomUUID();
    connectSocket(currentSessionId);
    chatTitleEl.textContent = text.replace(/\s+/g, " ").slice(0, 36) || "新对话";
  }
  addMessage("user", text);
  runningTimeline = freshTimeline();
  const assistant = addMessage("assistant", "", runningTimeline);
  currentTimelineEl = assistant.row.querySelector(".timeline");
  let assistantText = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId,
        target_paper_id: targetPaperId,
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`请求失败 (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const raw of events) {
        const eventName = (raw.match(/^event: (.+)$/m) || [])[1];
        const dataLine = (raw.match(/^data: (.+)$/m) || [])[1];
        if (!dataLine) continue;
        const data = JSON.parse(dataLine);
        if (eventName === "session") {
          currentSessionId = data.session_id;
          if (!socket || socket.readyState === WebSocket.CLOSED) {
            connectSocket(currentSessionId);
          }
        }
        if (eventName === "token") {
          assistantText += data.content;
          assistant.bubble.innerHTML = `<p>${renderMarkdown(assistantText)}</p>`;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        if (eventName === "done") {
          const finalTimeline = renderTimeline(data.timeline);
          if (currentTimelineEl) {
            currentTimelineEl.replaceWith(finalTimeline);
            currentTimelineEl = null;
          }
          if (data.evidence_report) {
            assistant.row.insertBefore(renderEvidence(data.evidence_report), assistant.bubble);
          }
        }
      }
    }
  } catch (error) {
    console.error("Chat request failed", error);
    assistant.bubble.innerHTML = "<p>网络或服务异常，未生成可引用的研究结论。请检查服务状态后重试。</p>";
  } finally {
    sendButton.disabled = false;
  }
  await loadSessions();
}

composerEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || sendButton.disabled) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  await sendMessage(text);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
});

newChatButton.addEventListener("click", () => {
  currentSessionId = null;
  chatTitleEl.textContent = "新对话";
  messagesEl.innerHTML = "";
  runningTimeline = freshTimeline();
  if (socket) socket.close();
  socketToken += 1;
  socket = null;
  connectionEl.textContent = "准备就绪";
  loadSessions();
});

loadSessions();

// ========== 论文库（跳转到 /papers 页面） ==========
// 从论文库页面跳转回来时，自动填充并发送提问
const pendingQuestion = localStorage.getItem("pendingPaperQuestion");
if (pendingQuestion) {
  const pendingPaperId = localStorage.getItem("pendingPaperId");
  localStorage.removeItem("pendingPaperQuestion");
  localStorage.removeItem("pendingPaperId");
  // 等 DOM 和 session 就绪后自动发送
  setTimeout(() => sendMessage(pendingQuestion, pendingPaperId), 300);
}

// ========== 论文上传 ==========
const uploadBtn = document.querySelector("#uploadBtn");
const fileInput = document.querySelector("#fileInput");
const uploadStatus = document.querySelector("#uploadStatus");
const uploadBatchPanel = document.querySelector("#uploadBatchPanel");
const uploadBatchSummary = document.querySelector("#uploadBatchSummary");
const uploadBatchJobs = document.querySelector("#uploadBatchJobs");
let uploadPollTimer = null;

uploadBtn.addEventListener("click", () => fileInput.click());

function stopUploadPolling() {
  if (uploadPollTimer) clearInterval(uploadPollTimer);
  uploadPollTimer = null;
}

function renderUploadBatch(batch) {
  uploadBatchPanel.style.display = "block";
  uploadBatchPanel.open = true;
  const failed = batch.jobs.filter((job) => job.status === "failed");
  uploadBatchSummary.textContent = `共 ${batch.total_count} 篇${failed.length ? ` · ${failed.length} 篇失败` : ""}`;
  uploadBatchJobs.innerHTML = batch.jobs.map((job) => `<div class="upload-job ${job.status}"><strong>${escapeHtml(job.filename)}</strong><span>${escapeHtml(job.stage_detail || job.status)}</span>${job.chunk_count ? `<small>${job.chunk_count} 个分块 · ${escapeHtml(job.parse_source || "")}</small>` : ""}${job.error ? `<small class="error">${escapeHtml(job.error)}</small>` : ""}</div>`).join("");
}

async function refreshUploadBatch(batchId) {
  const response = await fetch(`/api/upload-batches/${batchId}`);
  if (!response.ok) return stopUploadPolling();
  const batch = await response.json();
  renderUploadBatch(batch);
  if (!batch.jobs.some((job) => ["queued", "parsing", "chunking", "indexing"].includes(job.status))) stopUploadPolling();
}

function startUploadPolling(batchId) {
  stopUploadPolling();
  refreshUploadBatch(batchId);
  uploadPollTimer = setInterval(() => refreshUploadBatch(batchId), 1000);
}

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (!files.length) return;

  uploadStatus.style.display = "block";
  uploadStatus.className = "upload-status";
  uploadStatus.textContent = `正在加入队列: ${files.length} 篇 PDF...`;
  uploadBtn.disabled = true;

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  try {
    const res = await fetch("/api/uploads", { method: "POST", body: formData });
    const data = await res.json();
    if (res.ok) {
      uploadStatus.textContent = `✅ 已加入队列: ${data.accepted_count} 篇`;
      localStorage.setItem("paperAgentLastUploadBatch", data.batch_id);
      startUploadPolling(data.batch_id);
    } else {
      uploadStatus.className = "upload-status error";
      uploadStatus.textContent = `❌ ${data.detail || "上传失败"}`;
    }
  } catch (e) {
    uploadStatus.className = "upload-status error";
    uploadStatus.textContent = `❌ 网络错误: ${e.message}`;
  }

  uploadBtn.disabled = false;
  fileInput.value = "";
  setTimeout(() => { uploadStatus.style.display = "none"; }, 5000);
});

const lastUploadBatch = localStorage.getItem("paperAgentLastUploadBatch");
if (lastUploadBatch) startUploadPolling(lastUploadBatch);
