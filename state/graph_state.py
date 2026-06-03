"""
Paper Agent - LangGraph状态定义
定义多Agent协作的共享状态
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
import operator


class PaperInfo(TypedDict):
    """论文信息"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    status: str  # pending / parsed / chunked / embedded / indexed


class RetrievedChunk(TypedDict):
    """检索结果分块"""
    paper_arxiv_id: str
    paper_title: str
    chunk_index: int
    content: str
    score: float
    metadata: dict


class AgentState(TypedDict):
    """全局共享状态"""
    # 用户输入
    user_query: str
    search_query: Optional[str]  # 从用户输入中提取的搜索关键词

    # 消息历史（用于对话）
    messages: Annotated[list[BaseMessage], add_messages]

    # 论文相关
    target_papers: list[PaperInfo]          # 当前涉及的论文列表
    retrieved_chunks: list[RetrievedChunk]  # 检索到的相关分块

    # Agent输出
    search_results: Optional[dict]          # web搜索结果
    analysis: Optional[str]                 # 分析结论
    answer: Optional[str]                   # 最终回答

    # 流程控制
    next_agent: Optional[str]               # Supervisor决定的下一个Agent
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大轮次限制

    # 错误处理
    error: Optional[str]