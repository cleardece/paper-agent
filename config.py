import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger("paper-agent")

# LLM配置 - MiMo
LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Embedding模型 - 使用本地 BAAI/bge-m3 (SentenceTransformer)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # 留空自动检测 CUDA/CPU

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://host.docker.internal:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "paper_agent")

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "host.docker.internal")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Tavily搜索
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
def get_llm():
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