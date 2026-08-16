import logging
import os
import time
import threading
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("paper-agent")


def wait_for_gpu_release(timeout: int = 30, min_free_gb: float = None) -> bool:
    """等待 GPU 显存释放（MinerU 等外部进程释放后）再加载 embedding

    Args:
        timeout: 最长等待秒数
        min_free_gb: 需要的最小空闲显存（GB），None=自动根据硬件决定

    Returns:
        True=GPU 有足够空间，False=超时或不可用
    """
    if min_free_gb is None:
        from config import HW_TIER
        # BGE-M3 模型 ~2.3GB，加推理缓冲
        min_free_gb = {"high": 3.0, "medium": 3.5, "low": 4.0}[HW_TIER]
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        torch.cuda.empty_cache()
        for i in range(timeout):
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1024**3
            if free_gb >= min_free_gb:
                if i > 0:
                    logger.info(f"[GPU] 等待 {i}s 后显存释放完成，可用: {free_gb:.1f}GB")
                return True
            if i % 5 == 0:
                logger.info(f"[GPU] 等待显存释放... 可用: {free_gb:.1f}GB < {min_free_gb}GB ({i}/{timeout}s)")
            time.sleep(1)
        free, _ = torch.cuda.mem_get_info()
        logger.warning(f"[GPU] 等待超时，可用显存: {free/1024**3:.1f}GB")
        return False
    except Exception as e:
        logger.warning(f"[GPU] 检查显存失败: {e}")
        return False


def _find_model_path(model_name: str) -> str:
    """查找本地模型路径，找不到则返回 model_name 让 SentenceTransformer 下载"""
    candidates = [
        os.path.join("D:/huggingface_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"),
        os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return model_name


class EmbeddingService:
    def __init__(self, model_name: str = None, device: str = None):
        from config import EMBEDDING_MODEL, EMBEDDING_DEVICE
        self._model = None
        self._model_name = model_name or EMBEDDING_MODEL
        raw_device = device or EMBEDDING_DEVICE or ("cuda" if _cuda_available() else "cpu")
        self._device = raw_device.lower()  # PyTorch 只接受小写设备名（cuda, cpu）
        self._lock = threading.Lock()

    def _load_model(self, device: str = None):
        """加载模型到指定设备，失败则降级到 CPU"""
        target = (device or self._device).lower()  # PyTorch 只接受小写设备名
        if self._model is not None and self._device == target:
            return

        with self._lock:
            if self._model is not None and self._device == target:
                return

            model_path = _find_model_path(self._model_name)

            # GPU 模式：等待显存释放（MinerU 等外部进程释放）
            if target == "cuda":
                if not wait_for_gpu_release(timeout=30):
                    logger.warning("[Embedding] GPU 显存不足，降级到 CPU...")
                    target = "cpu"

            logger.info(f"[Embedding] 正在加载模型到 {target}...")

            try:
                self._model = SentenceTransformer(model_path, device=target)
                self._device = target
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and target == "cuda":
                    logger.warning(f"[Embedding] GPU 加载 OOM，降级到 CPU...")
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._model = SentenceTransformer(model_path, device="cpu")
                    self._device = "cpu"
                else:
                    raise

            import torch
            if self._device == "cuda" and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.info(f"[Embedding] GPU: {torch.cuda.get_device_name(0)}, "
                            f"显存: {total / 1024**3:.1f}GB, 可用: {free / 1024**3:.1f}GB")

            logger.info(f"[Embedding] 模型加载完成 (device={self._device})")

    def embed_texts(self, texts: list[str], batch_size: int = None,
                    normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()

        # 使用 config 中的硬件自适应 batch_size
        if batch_size is None:
            from config import EMBEDDING_BATCH_SIZE
            batch_size = EMBEDDING_BATCH_SIZE

        try:
            import torch
            with torch.no_grad():
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            if self._device == "cuda":
                torch.cuda.empty_cache()
            return embeddings.tolist()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self._device == "cuda":
                logger.warning("[Embedding] GPU 编码 OOM，降级到 CPU...")
                import torch
                torch.cuda.empty_cache()
                self._model = self._model.cpu()
                self._device = "cpu"
                return self.embed_texts(texts, batch_size=4, normalize=normalize)
            raise RuntimeError(f"Embedding failed: {e}")

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# 全局单例
embedding_service = EmbeddingService()