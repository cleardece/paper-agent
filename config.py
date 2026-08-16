import logging
import os
import platform
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger("paper-agent")


# ==================== 硬件自动检测 ====================
def _detect_hardware() -> dict:
    """检测系统硬件配置，返回适配参数"""
    info = {
        "ram_gb": 0,
        "cpu_cores": os.cpu_count() or 4,
        "gpu_name": None,
        "gpu_vram_gb": 0,
    }

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        # psutil 不可用时从系统命令获取
        try:
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                    capture_output=True, text=True, timeout=5
                )
                info["ram_gb"] = round(int(result.stdout.strip()) / (1024 ** 3), 1)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            info["ram_gb"] = round(kb / (1024 ** 2), 1)
                            break
        except Exception:
            info["ram_gb"] = 16  # 保守默认值

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except ImportError:
        pass

    return info


_hw = _detect_hardware()
_ram = _hw["ram_gb"]
_vram = _hw["gpu_vram_gb"]
_cores = _hw["cpu_cores"]

logger.info(f"[Hardware] RAM: {_ram}GB, CPU: {_cores} cores, GPU: {_hw['gpu_name']} ({_vram}GB VRAM)")

# ==================== 根据硬件自动配置 ====================

# 高配：>=32GB RAM 且 >=12GB VRAM
# 中配：16-31GB RAM 或 8-11GB VRAM
# 低配：<16GB RAM 且 <8GB VRAM
if _ram >= 32 and _vram >= 12:
    HW_TIER = "high"
elif _ram >= 16 or _vram >= 8:
    HW_TIER = "medium"
else:
    HW_TIER = "low"

logger.info(f"[Hardware] 硬件等级: {HW_TIER}")

# ==================== LLM ====================
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# ==================== Embedding ====================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # 留空自动检测 CUDA/CPU

# 根据硬件调整 embedding batch_size
# env 可覆盖：EMBEDDING_BATCH_SIZE=8
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "0"))
if not EMBEDDING_BATCH_SIZE:
    if HW_TIER == "high":
        EMBEDDING_BATCH_SIZE = 16
    elif HW_TIER == "medium":
        EMBEDDING_BATCH_SIZE = 8
    else:
        EMBEDDING_BATCH_SIZE = 2

# ==================== MongoDB ====================
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "paper_agent")

# ==================== Milvus ====================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# HNSW 索引参数：数据量大或高配时用更大参数
MILVUS_HNSW_M = int(os.getenv("MILVUS_HNSW_M", "0"))
MILVUS_HNSW_EF_CONSTRUCTION = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "0"))
MILVUS_SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "0"))
if not MILVUS_HNSW_M:
    if HW_TIER == "high":
        MILVUS_HNSW_M = 24
        MILVUS_HNSW_EF_CONSTRUCTION = 384
        MILVUS_SEARCH_EF = 128
    else:
        MILVUS_HNSW_M = 16
        MILVUS_HNSW_EF_CONSTRUCTION = 256
        MILVUS_SEARCH_EF = 64

# ==================== Neo4j（可选） ====================
NEO4J_URI = os.getenv("NEO4J_URI")  # bolt://localhost:7687
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==================== Reranker ====================
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ==================== PDF 解析 ====================
MINERU_URL = os.getenv("MINERU_URL")  # http://localhost:8888
# CPU 环境使用 pipeline；hybrid-auto-engine 需要可用的 vLLM/GPU。
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")

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

# ==================== 并发控制 ====================
# 低配机器限制并发，避免 OOM
MAX_CONCURRENT_PAPERS = int(os.getenv("MAX_CONCURRENT_PAPERS", "0"))
if not MAX_CONCURRENT_PAPERS:
    if HW_TIER == "high":
        MAX_CONCURRENT_PAPERS = 3   # 可以同时处理3篇
    elif HW_TIER == "medium":
        MAX_CONCURRENT_PAPERS = 1   # 串行处理
    else:
        MAX_CONCURRENT_PAPERS = 1


def get_llm():
    """获取 LLM 实例"""
    if not LLM_MODEL:
        raise ValueError("LLM_MODEL 未配置，请在 .env 中设置 LLM_MODEL")
    if not LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL 未配置，请在 .env 中设置 LLM_BASE_URL")
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
