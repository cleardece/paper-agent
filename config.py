import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger("paper-agent")

# ==================== LLM ====================
LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# ==================== Embedding ====================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # 留空自动检测 CUDA/CPU

# ==================== MongoDB ====================
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "paper_agent")

# ==================== Milvus ====================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# ==================== Neo4j（可选） ====================
NEO4J_URI = os.getenv("NEO4J_URI")  # bolt://localhost:7687
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==================== Reranker ====================
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ==================== PDF 解析 ====================
MINERU_URL = os.getenv("MINERU_URL")  # http://localhost:8888

# ==================== 搜索 API ====================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# ==================== MCP ====================
USE_MCP = os.getenv("USE_MCP", "true").lower() == "true"
# ArXiv MCP Server (Docker, SSE 模式)
MCP_ARXIV_URL = os.getenv("MCP_ARXIV_URL", "http://localhost:8050/sse")
# Semantic Scholar MCP (暂未实现)
MCP_SS_URL = os.getenv("MCP_SS_URL", "")

# ==================== LangSmith ====================
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")


def get_llm():
    """获取 LLM 实例"""
    logger.info(f"[LLM] 正在初始化 {LLM_MODEL} @ {LLM_BASE_URL}")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.3,
        timeout=30,
        max_retries=2,
    )
    logger.info("[LLM] 初始化完成")
    return llm
