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
    user_id: Optional[str]                # 用户ID（用于记忆）
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
    evidence_report: Optional[dict]         # 引用与检索证据的规则校验结果

    # 流程控制
    next_agent: Optional[str]               # Supervisor决定的下一个Agent
    iteration: int                          # 当前迭代轮次
    max_iterations: int                     # 最大轮次限制

    # Memory 相关
    reflection: Optional[dict]              # 反思记忆（insights, questions, etc.）
    session_id: Optional[str]               # 会话ID

    # 对话上下文
    conversation_context: Optional[str]     # 最近对话摘要（帮助理解跟随意图）
    conversation_summary: Optional[str]     # 较早对话的滚动摘要（不作为论文证据）
    research_profile_context: Optional[dict]  # 用户研究档案，仅用于理解意图，不作论文证据
    target_paper: Optional[str]             # 用户指代的论文标题（跟随意图用）
    target_paper_id: Optional[str]          # 论文库显式选择的稳定 arXiv/本地论文 ID
    resolved_paper_id: Optional[str]        # DirectAnalyzer 实际分析的论文 ID，供会话持久化
    active_paper_ids: list[str]             # 会话当前论文焦点，供追问消解使用
    active_section: Optional[str]           # 当前讨论章节
    active_task: Optional[str]              # 当前阅读/实验/写作任务
    open_questions: list[str]               # 会话尚未解决的问题

    # 错误处理
    error: Optional[str]
