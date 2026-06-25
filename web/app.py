"""FastAPI web entrypoint for Paper Agent."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

# 配置日志输出到 stderr，确保 uvicorn 能显示
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("paper-agent")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from config import get_llm
from state.graph_state import AgentState


AGENTS = ("supervisor", "translator", "fetcher", "retriever", "analyzer", "critic", "presenter")
QUESTION_FLOW = ("supervisor", "retriever", "analyzer", "critic", "presenter")
FETCH_FLOW = ("supervisor", "translator", "fetcher")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)
    timeline: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Session:
    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[ChatMessage] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.setdefault(session_id, set()).add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        sockets = self.active.get(session_id)
        if not sockets:
            return
        sockets.discard(websocket)
        if not sockets:
            self.active.pop(session_id, None)

    async def broadcast(self, session_id: str, payload: dict[str, Any]) -> None:
        sockets = list(self.active.get(session_id, set()))
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception as e:
                logger.warning(f"[WebSocket] 发送失败: {e}")
                self.disconnect(session_id, socket)


# Session 内存缓存（热数据）
sessions: dict[str, Session] = {}
manager = ConnectionManager()


def load_session_from_db(session_id: str) -> Session | None:
    """从 MongoDB 加载 Session"""
    from core.deps import get_container
    container = get_container()
    doc = container.mongodb.get_session(session_id)
    if not doc:
        return None

    session = Session(
        id=doc["session_id"],
        title=doc.get("title", ""),
        created_at=doc.get("created_at", time.time()),
        updated_at=doc.get("updated_at", time.time()),
    )

    # 加载消息
    messages_data = doc.get("messages", [])
    for msg in messages_data:
        session.messages.append(ChatMessage(
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            created_at=msg.get("created_at", time.time()),
            timeline=msg.get("timeline", []),
        ))

    return session


def save_session_to_db(session: Session):
    """保存 Session 到 MongoDB"""
    try:
        from core.deps import get_container
        container = get_container()
        messages_data = [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "timeline": msg.timeline,
            }
            for msg in session.messages
        ]
        container.mongodb.save_session(
            session_id=session.id,
            title=session.title,
            messages=messages_data,
            updated_at=session.updated_at,
        )
        logger.info(f"[Session] 已保存 Session: {session.id[:20]}...")
    except Exception as e:
        logger.error(f"[Session] 保存 Session 失败: {e}")


def summarize(value: Any, limit: int = 220) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def timeline_snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {
        agent: {
            "agent": agent,
            "status": "waiting",
            "detail": "",
            "input_summary": "",
            "output_summary": "",
            "retry_count": 0,
        }
        for agent in AGENTS
    }
    for event in events:
        agent = event.get("agent")
        if agent not in steps:
            continue
        steps[agent].update({k: v for k, v in event.items() if k in steps[agent]})
    ran = [event.get("agent") for event in events if event.get("status") in {"running", "completed"}]
    expected = FETCH_FLOW if "fetcher" in ran else QUESTION_FLOW
    for agent, step in steps.items():
        if agent not in expected and step["status"] == "waiting":
            step["status"] = "skipped"
    return [steps[agent] for agent in AGENTS]


async def push_status(session: Session, payload: dict[str, Any]) -> None:
    event = {
        "type": "agent_status",
        "timestamp": time.time(),
        **payload,
    }
    session.events.append(event)
    await manager.broadcast(session.id, event)
    # 打印到控制台方便调试
    logger.info(f"[{payload.get('agent', '?')}] {payload.get('status', '?')}: {payload.get('detail', '')}")


def run_sync_or_async(fn: Callable[[AgentState], Any], state: AgentState) -> Awaitable[dict[str, Any]]:
    async def runner() -> dict[str, Any]:
        result = fn(state)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    return runner()


def wrap_agent(name: str, invoke: Callable[[AgentState], Any], session: Session) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def wrapped(state: AgentState) -> dict[str, Any]:
        retry_count = int(state.get("iteration", 0)) if name == "critic" else 0
        await push_status(
            session,
            {
                "agent": name,
                "status": "running",
                "detail": agent_running_detail(name, retry_count),
                "input_summary": summarize(state),
                "retry_count": retry_count,
            },
        )
        try:
            result = await run_sync_or_async(invoke, state)
            await push_status(
                session,
                {
                    "agent": name,
                    "status": "completed",
                    "detail": agent_done_detail(name, result),
                    "output_summary": summarize(result),
                    "retry_count": int(result.get("iteration", retry_count)) if name == "critic" else retry_count,
                },
            )
            return result
        except Exception as exc:
            logger.error(f"[{name}] 执行失败: {exc}", exc_info=True)
            await push_status(
                session,
                {
                    "agent": name,
                    "status": "error",
                    "detail": f"{name} 执行失败：{exc}",
                    "output_summary": summarize({"error": str(exc)}),
                    "retry_count": retry_count,
                },
            )
            # 返回错误状态，不 raise，让 workflow 继续执行
            return {"error": f"{name} 执行失败: {exc}"}

    return wrapped


def agent_running_detail(name: str, retry_count: int = 0) -> str:
    details = {
        "supervisor": "正在分析用户意图...",
        "translator": "正在翻译查询关键词...",
        "fetcher": "正在检索并入库论文...",
        "retriever": "正在检索相关论文片段...",
        "analyzer": "正在综合分析证据...",
        "critic": f"正在评估回答质量...（第 {retry_count + 1} 次）",
        "presenter": "正在整理最终回复...",
    }
    return details.get(name, f"{name} 正在执行...")


def agent_done_detail(name: str, result: dict[str, Any]) -> str:
    if name == "supervisor":
        return f"路由到 {result.get('next_agent', 'END')}"
    if name == "critic":
        score = result.get("critic_score", {}).get("score", "N/A")
        return f"质量评分 {score}，下一步 {result.get('next_agent', 'END')}"
    if result.get("error"):
        return str(result["error"])
    return "已完成"


def create_web_initial_state(query: str) -> AgentState:
    return {
        "user_query": query,
        "search_query": None,
        "messages": [],
        "target_papers": [],
        "retrieved_chunks": [],
        "search_results": None,
        "analysis": None,
        "answer": None,
        "critic_score": None,
        "next_agent": None,
        "iteration": 0,
        "max_iterations": 2,
        "error": None,
    }


def supervisor_route(state: AgentState) -> str:
    next_agent = state.get("next_agent", "END")
    if state.get("error"):
        logger.warning("[SupervisorRoute] 检测到错误，路由到 presenter")
        return "presenter"
    if next_agent == "fetcher":
        # fetcher 需要先经过 translator 翻译查询
        return "translator"
    if next_agent not in ("retriever", "END"):
        logger.warning(f"[SupervisorRoute] 非法路由 '{next_agent}'，使用 END")
        return "END"
    return next_agent


def fetcher_route(state: AgentState) -> str:
    # fetcher 成功入库后希望继续检索分析
    if state.get("next_agent") == "retriever":
        # 清除 fetcher 的入库摘要，避免覆盖后续 retriever 的分析结果
        state["answer"] = None
        return "retriever"
    # 如果 fetcher 直接返回了 answer（如没有 PDF 时），路由到 presenter 展示
    if state.get("answer"):
        return "presenter"
    return END


def critic_route(state: AgentState) -> str:
    # 如果有错误，强制终止循环
    if state.get("error"):
        logger.warning("[CriticRoute] 检测到错误，强制终止循环")
        return "presenter"
    next_agent = state.get("next_agent", "END")
    # 如果 next_agent 不在合法范围内，强制终止
    if next_agent not in ("presenter", "retriever", "END"):
        logger.warning(f"[CriticRoute] 非法路由 '{next_agent}'，强制终止")
        return "presenter"
    return next_agent


def build_traced_workflow(session: Session):
    from core.deps import get_container

    # 获取单例服务容器
    container = get_container()
    agents = container.create_agents()

    graph = StateGraph(AgentState)
    for name, agent in agents.items():
        graph.add_node(name, wrap_agent(name, agent.invoke, session))

    # 构建图结构
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", supervisor_route, {
        "translator": "translator",
        "retriever": "retriever",
        "END": "presenter",
    })
    graph.add_edge("translator", "fetcher")
    graph.add_conditional_edges("fetcher", fetcher_route, {"presenter": "presenter", "retriever": "retriever", END: END})
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")
    graph.add_conditional_edges("critic", critic_route, {
        "presenter": "presenter",
        "retriever": "retriever",
        "END": END
    })

    # Presenter → Reflector → END（如果有 reflector）
    if "reflector" in agents:
        graph.add_edge("presenter", "reflector")
        graph.add_edge("reflector", END)
    else:
        graph.add_edge("presenter", END)

    return graph.compile()


def get_or_create_session(session_id: str | None, message: str = "") -> Session:
    if session_id:
        # 先查内存缓存
        if session_id in sessions:
            return sessions[session_id]
        # 再查 MongoDB
        session = load_session_from_db(session_id)
        if session:
            sessions[session_id] = session
            return session

    # 创建新 Session
    new_id = session_id or uuid4().hex
    title = message.strip().replace("\n", " ")[:36] or "新对话"
    session = Session(id=new_id, title=title)
    sessions[new_id] = session
    return session


def serialize_session(session: Session, include_messages: bool = False) -> dict[str, Any]:
    data = {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
    if include_messages:
        data["messages"] = [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "timeline": msg.timeline,
            }
            for msg in session.messages
        ]
    return data


app = FastAPI(title="Paper Agent Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    from core.deps import get_container
    # 初始化服务容器（单例）
    get_container()
    logger.info("=" * 60)
    logger.info("Paper Agent Web 启动成功!")
    logger.info("访问 http://localhost:8000 开始使用")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    from core.deps import close_container
    close_container()
    logger.info("Paper Agent Web 已关闭")


static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/papers")
async def papers_page() -> FileResponse:
    return FileResponse(static_dir / "papers.html")


@app.get("/api/test")
async def test_connection():
    """测试后端组件连接"""
    from core.deps import get_container
    container = get_container()
    results = {
        "mongodb": "OK" if container.mongodb else "NOT INITIALIZED",
        "milvus": "OK" if container.milvus else "NOT INITIALIZED",
        "llm": "OK" if container.llm else "NOT INITIALIZED",
        "embedder": "OK" if container.embedder else "NOT INITIALIZED",
    }
    return results


import asyncio

@app.post("/api/upload")
async def upload_paper(file: UploadFile):
    """上传本地 PDF 论文，后台异步解析并入库"""
    import json
    import os
    from core.deps import get_container

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    container = get_container()

    # 保存到临时目录
    tmp_dir = "tmp_pdfs"
    os.makedirs(tmp_dir, exist_ok=True)
    pdf_path = os.path.join(tmp_dir, file.filename)

    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)

    logger.info(f"[Upload] 收到文件: {file.filename} ({len(content)} bytes)")

    # 生成 arxiv_id
    arxiv_id = os.path.splitext(file.filename)[0].replace(" ", "_").replace("/", "_")
    if len(arxiv_id) > 60:
        arxiv_id = arxiv_id[:60]

    # 立即返回，后台处理
    asyncio.create_task(_process_upload(container, pdf_path, file.filename, arxiv_id, session_id=None))

    return {
        "message": "上传成功，正在后台解析...",
        "arxiv_id": arxiv_id,
        "title": file.filename,
    }


async def _process_upload(container, pdf_path, filename, arxiv_id, session_id=None):
    """后台处理上传的 PDF"""
    import json
    from core.cache import cache

    try:
        # 检查缓存（会话级）
        cache_key = f"paper:{arxiv_id}"
        cached = cache.get(session_id, cache_key)
        if cached:
            logger.info(f"[Upload] 使用缓存: {arxiv_id}")
            chunks = cached["chunks"]
            result = cached.get("result", {})
        else:
            # 1. 解析 PDF
            result = await asyncio.get_event_loop().run_in_executor(
                None, container.pdf_parser.parse, pdf_path
            )
            logger.info(f"[Upload] 解析完成: {result['source']}, {len(result['sections'])} 章节")

            # 2. 分块
            chunks = container.pdf_parser.chunk(result["sections"])
            logger.info(f"[Upload] 分块完成: {len(chunks)} 个")

            # 缓存解析和分块结果（会话级）
            cache.set(session_id, cache_key, {"chunks": chunks, "result": result})

        # 3. 存 MongoDB
        container.mongodb.upsert_paper({
            "arxiv_id": arxiv_id,
            "title": result.get("title", filename),
            "abstract": "",
            "authors": [],
            "pdf_url": f"local://{filename}",
            "status": "chunked",
        })
        mongo_chunks = [
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
        ]
        container.mongodb.insert_chunks(mongo_chunks)
        logger.info(f"[Upload] MongoDB 入库完成")

        # 4. Embedding + Milvus
        texts = [c["content"] for c in chunks]
        vectors = await asyncio.get_event_loop().run_in_executor(
            None, container.embedder.embed_texts, texts
        )
        milvus_records = [
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "embedding": vectors[i],
                "metadata_json": json.dumps(c.get("metadata", {})),
            }
            for i, c in enumerate(chunks)
        ]
        container.milvus.insert(milvus_records)
        logger.info(f"[Upload] Milvus 入库完成: {len(chunks)} 分块")

        # 5. 更新状态为 indexed
        container.mongodb.update_paper_status(arxiv_id, "indexed")
        logger.info(f"[Upload] 论文处理完成: {filename}")

    except Exception as e:
        logger.error(f"[Upload] 后台处理失败: {e}", exc_info=True)


# ==================== 论文列表 API ====================

@app.get("/api/papers")
async def list_papers():
    """列出所有论文"""
    from core.deps import get_container
    container = get_container()
    papers = container.mongodb.list_papers(limit=50)
    return [
        {
            "arxiv_id": p.get("arxiv_id", ""),
            "title": p.get("title", ""),
            "status": p.get("status", ""),
        }
        for p in papers
    ]


@app.delete("/api/papers/{arxiv_id}")
async def delete_paper(arxiv_id: str):
    """删除单篇论文（含 Milvus 向量）"""
    from core.deps import get_container
    container = get_container()
    container.mongodb.delete_paper(arxiv_id)
    container.milvus.delete_by_paper(arxiv_id)
    return {"message": "deleted"}


@app.post("/api/compare")
async def compare_papers(body: dict):
    """对比多篇论文"""
    from core.deps import get_container
    container = get_container()

    paper_ids = body.get("paper_ids", [])
    if len(paper_ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要2篇论文进行对比")

    # 检索每篇论文的内容
    all_chunks = []
    paper_info = []
    for pid in paper_ids:
        chunks = container.mongodb.get_chunks_by_paper(pid)
        paper = container.mongodb.get_paper(pid)
        title = paper.get("title", pid) if paper else pid
        paper_info.append({"arxiv_id": pid, "title": title, "chunks": len(chunks)})
        for c in chunks:
            all_chunks.append({
                "paper_title": title,
                "chunk_index": c.get("chunk_index", 0),
                "content": c.get("content", ""),
                "score": 1.0,
            })

    if not all_chunks:
        raise HTTPException(status_code=404, detail="未找到论文内容")

    # 用 Analyzer 分析
    from agents.analyzer import AnalyzerAgent, _build_analyzer_tools
    tools = _build_analyzer_tools(container.mongodb)
    analyzer = AnalyzerAgent(container.llm, container.mongodb)

    state = {
        "user_query": f"对比分析以下论文：{', '.join(p['title'][:30] for p in paper_info)}",
        "retrieved_chunks": all_chunks[:20],  # 限制 chunk 数量
    }

    try:
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(
            None, analyzer.invoke, state
        )
        return {
            "papers": paper_info,
            "analysis": result.get("analysis", ""),
        }
    except Exception as e:
        logger.error(f"[Compare] 对比失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对比失败: {str(e)}")


@app.get("/api/citations/{arxiv_id}")
async def get_citations(arxiv_id: str):
    """获取论文的引用关系"""
    from core.deps import get_container
    container = get_container()

    # 从 Semantic Scholar 获取引用数据
    try:
        import httpx
        # 尝试用 arxiv_id 搜索 Semantic Scholar
        response = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}",
            params={"fields": "title,references.title,references.paperId,citations.title,citations.paperId"},
            headers={"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            references = data.get("references", []) or []
            citations = data.get("citations", []) or []
            return {
                "paper_id": arxiv_id,
                "title": data.get("title", ""),
                "references": [{"title": r.get("title", ""), "paper_id": r.get("paperId", "")} for r in references[:20]],
                "citations": [{"title": c.get("title", ""), "paper_id": c.get("paperId", "")} for c in citations[:20]],
            }
        else:
            return {"paper_id": arxiv_id, "references": [], "citations": [], "error": f"API 返回 {response.status_code}"}
    except Exception as e:
        logger.error(f"[Citations] 获取引用失败: {e}")
        return {"paper_id": arxiv_id, "references": [], "citations": [], "error": str(e)}


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    from core.deps import get_container
    container = get_container()
    # 从 MongoDB 加载所有 Session
    db_sessions = container.mongodb.list_sessions(limit=100)
    result = []
    for doc in db_sessions:
        result.append({
            "id": doc.get("session_id", ""),
            "title": doc.get("title", ""),
            "created_at": doc.get("created_at", 0),
            "updated_at": doc.get("updated_at", 0),
        })
    return sorted(result, key=lambda item: item["updated_at"], reverse=True)


# 定期清理过期 Session
def maybe_cleanup_sessions():
    """每 100 次请求清理一次"""
    if not hasattr(maybe_cleanup_sessions, "counter"):
        maybe_cleanup_sessions.counter = 0
    maybe_cleanup_sessions.counter += 1
    if maybe_cleanup_sessions.counter >= 100:
        maybe_cleanup_sessions.counter = 0
        # 清理 MongoDB 中超过 24 小时的 Session
        from core.deps import get_container
        container = get_container()
        # 简单实现：不做清理，只清内存缓存
        sessions.clear()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = sessions.get(session_id)
    if not session:
        # 尝试从 MongoDB 加载
        session = load_session_from_db(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        sessions[session_id] = session
    return serialize_session(session, include_messages=True)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """删除单个 Session 及其缓存"""
    from core.deps import get_container
    from core.cache import cache
    container = get_container()
    container.mongodb.delete_session(session_id)
    cache.delete_session(session_id)
    sessions.pop(session_id, None)
    return {"message": "deleted"}


@app.delete("/api/sessions")
async def delete_all_sessions() -> dict[str, Any]:
    """删除所有 Session 及其缓存"""
    from core.deps import get_container
    from core.cache import cache
    container = get_container()
    count = container.mongodb.delete_all_sessions()
    cache.delete_all()
    sessions.clear()
    return {"message": f"deleted {count} sessions"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    # 定期清理过期 Session
    maybe_cleanup_sessions()

    session = get_or_create_session(request.session_id, request.message)
    session.updated_at = time.time()
    session.events = []
    session.messages.append(ChatMessage(role="user", content=request.message))
    # 持久化用户消息
    save_session_to_db(session)

    async def event_stream():
        yield f"event: session\ndata: {json.dumps({'session_id': session.id}, ensure_ascii=False)}\n\n"
        answer = None
        try:
            logger.info("[Chat] 开始构建 workflow...")
            workflow = build_traced_workflow(session)
            state = create_web_initial_state(request.message)
            logger.info(f"[Chat] 开始执行 workflow，查询: {request.message[:50]}...")
            result = await asyncio.wait_for(workflow.ainvoke(state), timeout=300)
            answer = result.get("answer") or result.get("error") or "未生成回复。"
            logger.info(f"[Chat] Workflow 执行完成")
        except asyncio.TimeoutError:
            answer = "抱歉，处理时间较长，请稍后重试。"
            logger.warning("[Chat] Workflow 执行超时")
        except Exception as exc:
            answer = f"抱歉，处理过程中出现问题，请稍后重试。"
            logger.error(f"[Chat] Workflow 执行失败: {exc}", exc_info=True)

        if answer is None:
            answer = "抱歉，未能生成回复，请稍后重试。"

        timeline = timeline_snapshot(session.events)
        session.messages.append(ChatMessage(role="assistant", content=answer, timeline=timeline))
        session.updated_at = time.time()
        save_session_to_db(session)

        # 流式输出最终回答：逐句发送，模拟打字效果
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', answer)
        for sentence in sentences:
            if not sentence.strip():
                continue
            yield f"event: token\ndata: {json.dumps({'content': sentence}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)

        yield f"event: done\ndata: {json.dumps({'timeline': timeline}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
