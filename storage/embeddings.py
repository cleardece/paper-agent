import threading
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self._model = None
        self._model_name = model_name
        self._device = device or ("cuda" if _cuda_available() else "cpu")
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device
                    )

    def embed_texts(self, texts: list[str], batch_size: int = 32,
                    normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
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