"""
Paper Agent - 依赖注入模块
单例模式管理所有基础设施连接
"""

import logging
import os
from config import get_llm
from storage.mongodb import MongoDBClient
from storage.milvus import MilvusClient
from tools.semantic_scholar import SemanticScholarAPI
from tools.arxiv_api import ArxivAPI
from tools.embeddings import EmbeddingService
from tools.pdf_parser import PDFParser
from tools.code_generator import CodeGenerator

logger = logging.getLogger("paper-agent")


class ServiceContainer:
    """服务容器 - 单例管理所有基础设施"""

    def __init__(self):
        logger.info("[Container] 正在初始化服务容器...")

        # 基础设施
        self.llm = get_llm()
        self.mongodb = MongoDBClient()
        self.milvus = MilvusClient()
        self.embedder = EmbeddingService()

        # 优先使用 Semantic Scholar，fallback 到 arXiv
        ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if ss_api_key:
            self.paper_search = SemanticScholarAPI(api_key=ss_api_key)
            logger.info("[Container] 使用 Semantic Scholar API")
        else:
            self.paper_search = ArxivAPI()
            logger.info("[Container] 使用 arXiv API（无 Semantic Scholar API Key）")

        self.pdf_parser = PDFParser()
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

        return {
            "supervisor": SupervisorAgent(self.llm),
            "fetcher": FetcherAgent(
                self.paper_search, self.pdf_parser, self.mongodb,
                self.embedder, self.milvus
            ),
            "retriever": RetrieverAgent(
                self.embedder, self.milvus, self.mongodb
            ),
            "analyzer": AnalyzerAgent(self.llm, self.mongodb),
            "critic": CriticAgent(self.llm),
            "presenter": PresenterAgent(self.llm, self.code_generator),
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
