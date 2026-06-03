"""FastAPI web entrypoint for Paper Agent."""

from __future__ import annotations

import asyncio
import html
import json
import logging
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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


sessions: dict[str, Session] = {}
manager = ConnectionManager()

# Session 过期时间（秒）
SESSION_TTL = 3600  # 1小时


def cleanup_expired_sessions():
    """清理过期的 Session"""
    now = time.time()
    expired_ids = [
        sid for sid, session in sessions.items()
        if now - session.updated_at > SESSION_TTL
    ]
    for sid in expired_ids:
        del sessions[sid]
        logger.info(f"[Session] 已清理过期 Session: {sid}")

    if expired_ids:
        logger.info(f"[Session] 共清理 {len(expired_ids)} 个过期 Session")


# 定期清理（在每次请求时触发）
def maybe_cleanup_sessions():
    """每 100 次请求清理一次"""
    if not hasattr(maybe_cleanup_sessions, "counter"):
        maybe_cleanup_sessions.counter = 0
    maybe_cleanup_sessions.counter += 1
    if maybe_cleanup_sessions.counter >= 100:
        maybe_cleanup_sessions.counter = 0
        cleanup_expired_sessions()


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
        "translator": "正在翻译查询...",
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
    if name == "translator":
        search_query = result.get("search_query", "")
        return f"翻译为: {search_query[:30]}..."
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
        "max_iterations": 3,
        "error": None,
    }


def supervisor_route(state: AgentState) -> str:
    next_agent = state.get("next_agent", "END")
    # 如果是 fetcher，先经过翻译
    if next_agent == "fetcher":
        return "translator"
    return next_agent


def fetcher_route(state: AgentState) -> str:
    return END


def critic_route(state: AgentState) -> str:
    # 如果有错误，强制终止循环
    if state.get("error"):
        logger.warning("[CriticRoute] 检测到错误，强制终止循环")
        return "presenter"
    return state.get("next_agent", "END")


def build_traced_workflow(session: Session):
    from core.deps import get_container
    from agents.translator import TranslatorAgent

    # 获取单例服务容器
    container = get_container()
    agents = container.create_agents()

    # 添加翻译 Agent
    translator = TranslatorAgent(container.llm)
    agents["translator"] = translator

    graph = StateGraph(AgentState)
    for name, agent in agents.items():
        graph.add_node(name, wrap_agent(name, agent.invoke, session))

    # 构建图结构
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", supervisor_route, {
        "fetcher": "translator",  # fetcher 先经过翻译
        "retriever": "retriever",
        "END": "presenter"
    })
    graph.add_edge("translator", "fetcher")  # 翻译后传给 fetcher
    graph.add_conditional_edges("fetcher", fetcher_route, {END: END})
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")
    graph.add_conditional_edges("critic", critic_route, {
        "presenter": "presenter",
        "retriever": "retriever",
        "END": END
    })
    graph.add_edge("presenter", END)
    return graph.compile()


def get_or_create_session(session_id: str | None, message: str = "") -> Session:
    if session_id and session_id in sessions:
        return sessions[session_id]
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


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return sorted((serialize_session(s) for s in sessions.values()), key=lambda item: item["updated_at"], reverse=True)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return serialize_session(session, include_messages=True)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    # 定期清理过期 Session
    maybe_cleanup_sessions()

    session = get_or_create_session(request.session_id, request.message)
    session.updated_at = time.time()
    session.events = []
    session.messages.append(ChatMessage(role="user", content=request.message))

    async def event_stream():
        yield f"event: session\ndata: {json.dumps({'session_id': session.id}, ensure_ascii=False)}\n\n"
        answer = None
        try:
            logger.info("[Chat] 开始构建 workflow...")
            workflow = build_traced_workflow(session)
            state = create_web_initial_state(request.message)
            logger.info(f"[Chat] 开始执行 workflow，查询: {request.message[:50]}...")
            result = await asyncio.wait_for(workflow.ainvoke(state), timeout=120)
            answer = result.get("answer") or result.get("error") or "未生成回复。"
            logger.info(f"[Chat] Workflow 执行完成")
        except asyncio.TimeoutError:
            answer = "执行超时（120秒），请检查服务连接或稍后重试。"
            logger.warning("[Chat] Workflow 执行超时")
        except Exception as exc:
            answer = f"执行失败：{html.escape(str(exc))}"
            logger.error(f"[Chat] Workflow 执行失败: {exc}", exc_info=True)

        if answer is None:
            answer = "未知错误，未生成回复。"

        timeline = timeline_snapshot(session.events)
        session.messages.append(ChatMessage(role="assistant", content=answer, timeline=timeline))
        session.updated_at = time.time()
        for i in range(0, len(answer), 48):
            chunk = answer[i : i + 48]
            yield f"event: token\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
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
