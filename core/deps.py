"""
Paper Agent - 依赖注入模块
单例模式管理所有基础设施连接
"""

import logging
import os

from config import (
    MINERU_OFFICIAL_BASE_URL,
    MINERU_OFFICIAL_POLL_SECONDS,
    MINERU_OFFICIAL_TIMEOUT_SECONDS,
    MINERU_OFFICIAL_TOKEN,
    SEMANTIC_SCHOLAR_API_KEY,
    get_llm,
)
from storage.mongodb import MongoDBClient
from storage.research_memory import ResearchMemoryService
from storage.milvus import MilvusClient
from tools.embeddings import EmbeddingService
from tools.pdf_parser import PDFParser
from tools.mineru_official import OfficialMinerUClient
from tools.code_generator import CodeGenerator

logger = logging.getLogger("paper-agent")


def _create_pdf_parser() -> PDFParser:
    """构造唯一的 MinerU 官方精准 API 解析器。"""
    official_client = OfficialMinerUClient(
        token=MINERU_OFFICIAL_TOKEN,
        base_url=MINERU_OFFICIAL_BASE_URL,
        poll_seconds=MINERU_OFFICIAL_POLL_SECONDS,
        timeout_seconds=MINERU_OFFICIAL_TIMEOUT_SECONDS,
    )
    parser = PDFParser(official_client=official_client)
    logger.info("[Container] 论文解析器: %s", parser.provider_label)
    return parser


def _create_paper_search():
    """创建论文搜索服务 - 优先 ArXiv MCP，降级到直接 API"""
    # 检查是否启用 MCP
    use_mcp = os.getenv("USE_MCP", "true").lower() == "true"

    if use_mcp:
        # 优先 ArXiv MCP（已验证可用）
        try:
            from config import MCP_ARXIV_URL
            if MCP_ARXIV_URL:
                from tools.mcp_arxiv import ArxivAPI
                logger.info("[Container] 使用 ArXiv MCP")
                return ArxivAPI(use_mcp=True)
        except Exception as e:
            logger.warning(f"[Container] ArXiv MCP 失败: {e}")

        # 降级到 Semantic Scholar MCP
        try:
            from config import MCP_SS_URL
            if MCP_SS_URL:
                from tools.mcp_semantic_scholar import SemanticScholarAPI
                logger.info("[Container] 使用 Semantic Scholar MCP")
                return SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY, use_mcp=True)
        except Exception as e:
            logger.warning(f"[Container] Semantic Scholar MCP 失败: {e}")

    # 降级到直接 API
    if SEMANTIC_SCHOLAR_API_KEY:
        from tools.semantic_scholar import SemanticScholarAPI
        logger.info("[Container] 使用 Semantic Scholar 直接 API")
        return SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY)
    else:
        from tools.arxiv_api import ArxivAPI
        logger.info("[Container] 使用 arXiv 直接 API")
        return ArxivAPI()


class ServiceContainer:
    """服务容器 - 单例管理所有基础设施"""

    def __init__(self):
        logger.info("[Container] 正在初始化服务容器...")

        # 先校验官方 MinerU 配置，缺少 Token 时不要继续连接其他基础设施。
        self.pdf_parser = _create_pdf_parser()

        # 基础设施
        self.llm = get_llm()
        self.mongodb = MongoDBClient()
        self.milvus = MilvusClient()
        self.embedder = EmbeddingService()
        # 持久化研究图谱：不依赖可选 Neo4j，也不会替代已有 RAG。
        from storage.research_graph import ResearchGraphRepository
        self.research_graph = ResearchGraphRepository(self.mongodb.db)

        # 论文搜索（MCP 优先）
        self.paper_search = _create_paper_search()

        # Hybrid Search
        from tools.hybrid_search import HybridSearch
        self.hybrid_search = HybridSearch()

        # Knowledge Graph
        from storage.knowledge_graph import KnowledgeGraph
        self.knowledge_graph = KnowledgeGraph(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
        )

        # Memory 系统
        self.mongodb.memory.llm = self.llm
        self.mongodb.memory.kg = self.knowledge_graph
        self.mongodb.user_memory.llm = self.llm
        self.research_memory = ResearchMemoryService(self.mongodb.db, self.llm)

        self.code_generator = CodeGenerator(self.llm)

        logger.info("[Container] 服务容器初始化完成")

    def create_agents(self):
        """创建所有 Agent 实例"""
        from agents.supervisor import SupervisorAgent
        from agents.fetcher import FetcherAgent
        from agents.retriever import RetrieverAgent
        from agents.direct_analyzer import DirectAnalyzerAgent
        from agents.analyzer import AnalyzerAgent
        from agents.critic import CriticAgent
        from agents.presenter import PresenterAgent
        from agents.reflector import ReflectorAgent
        from core.paper_context import PaperContextResolver
        from core.turn_context import TurnContextBuilder

        return {
            "paper_context_resolver": PaperContextResolver(
                self.mongodb, self.llm
            ),
            "supervisor": SupervisorAgent(self.llm, self.mongodb),
            "turn_context": TurnContextBuilder(self.mongodb),
            "fetcher": FetcherAgent(
                self.paper_search, self.pdf_parser, self.mongodb,
                self.embedder, self.milvus
            ),
            "retriever": RetrieverAgent(
                self.embedder, self.milvus, self.mongodb, self.llm,
                self.hybrid_search, self.research_graph
            ),
            "direct_analyzer": DirectAnalyzerAgent(
                self.llm, self.mongodb, self.embedder, self.milvus,
                self.pdf_parser
            ),
            "analyzer": AnalyzerAgent(self.llm, self.mongodb),
            "critic": CriticAgent(self.llm),
            "presenter": PresenterAgent(self.llm, self.code_generator),
            "reflector": ReflectorAgent(self.llm),
        }

    def close(self):
        """关闭所有连接"""
        logger.info("[Container] 正在关闭服务连接...")
        try:
            self.mongodb.close()
        except Exception as e:
            logger.error(f"[Container] 关闭 MongoDB 失败: {e}")
        try:
            self.milvus.close()
        except Exception as e:
            logger.error(f"[Container] 关闭 Milvus 失败: {e}")
        try:
            self.knowledge_graph.close()
        except Exception as e:
            logger.error(f"[Container] 关闭 Knowledge Graph 失败: {e}")
        logger.info("[Container] 服务连接已关闭")


# 全局单例
_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """获取服务容器单例"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def close_container():
    """关闭服务容器"""
    global _container
    if _container is not None:
        _container.close()
        _container = None
