"""
Paper Agent - 依赖注入模块
单例模式管理所有基础设施连接
"""

import logging
import os
from config import get_llm, MINERU_URL, SEMANTIC_SCHOLAR_API_KEY
from storage.mongodb import MongoDBClient
from storage.milvus import MilvusClient
from tools.embeddings import EmbeddingService
from tools.pdf_parser import PDFParser
from tools.code_generator import CodeGenerator

logger = logging.getLogger("paper-agent")


def _create_paper_search():
    """创建论文搜索服务 - 优先 MCP，降级到直接 API"""
    # 检查是否启用 MCP
    use_mcp = os.getenv("USE_MCP", "true").lower() == "true"

    if use_mcp:
        try:
            from tools.mcp_semantic_scholar import SemanticScholarAPI
            logger.info("[Container] 尝试使用 Semantic Scholar MCP...")
            return SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY, use_mcp=True)
        except Exception as e:
            logger.warning(f"[Container] Semantic Scholar MCP 失败: {e}")

        try:
            from tools.mcp_arxiv import ArxivAPI
            logger.info("[Container] 尝试使用 ArXiv MCP...")
            return ArxivAPI(use_mcp=True)
        except Exception as e:
            logger.warning(f"[Container] ArXiv MCP 失败: {e}")

    # 降级到直接 API
    if SEMANTIC_SCHOLAR_API_KEY:
        from tools.semantic_scholar import SemanticScholarAPI
        logger.info("[Container] 使用 Semantic Scholar 直接 API")
        return SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY, use_mcp=False)
    else:
        from tools.arxiv_api import ArxivAPI
        logger.info("[Container] 使用 arXiv 直接 API")
        return ArxivAPI(use_mcp=False)


class ServiceContainer:
    """服务容器 - 单例管理所有基础设施"""

    def __init__(self):
        logger.info("[Container] 正在初始化服务容器...")

        # 基础设施
        self.llm = get_llm()
        self.mongodb = MongoDBClient()
        self.milvus = MilvusClient()
        self.embedder = EmbeddingService()

        # 论文搜索（MCP 优先）
        self.paper_search = _create_paper_search()

        # PDF 解析：优先 MinerU，fallback 到 pdfplumber
        self.pdf_parser = PDFParser(mineru_url=MINERU_URL)
        if MINERU_URL:
            logger.info(f"[Container] 使用 MinerU: {MINERU_URL}")
        else:
            logger.info("[Container] 使用 pdfplumber（未配置 MinerU）")

        # Hybrid Search
        from tools.hybrid_search import HybridSearch
        self.hybrid_search = HybridSearch()

        # Reranker
        from tools.reranker import Reranker
        self.reranker = Reranker()

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

        self.code_generator = CodeGenerator(self.llm)

        logger.info("[Container] 服务容器初始化完成")

    def create_agents(self):
        """创建所有 Agent 实例"""
        from agents.supervisor import SupervisorAgent
        from agents.fetcher import FetcherAgent
        from agents.retriever import RetrieverAgent
        from agents.analyzer import AnalyzerAgent
        from agents.critic import CriticAgent
        from agents.presenter import PresenterAgent
        from agents.reflector import ReflectorAgent

        return {
            "supervisor": SupervisorAgent(self.llm),
            "fetcher": FetcherAgent(
                self.paper_search, self.pdf_parser, self.mongodb,
                self.embedder, self.milvus
            ),
            "retriever": RetrieverAgent(
                self.embedder, self.milvus, self.mongodb, self.llm
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
