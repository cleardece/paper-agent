"""
Paper Agent - 缓存模块
论文搜索结果和 Embedding 缓存
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("paper-agent")


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Args:
            cache_dir: 缓存目录
            ttl: 缓存过期时间（秒），默认1小时
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self._memory_cache = {}  # 内存缓存

    def get(self, key: str) -> dict | None:
        """获取缓存"""
        # 先查内存缓存
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.info(f"[Cache] 内存缓存命中: {key[:30]}...")
                return entry["data"]
            else:
                del self._memory_cache[key]

        # 再查文件缓存
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] < self.ttl:
                    logger.info(f"[Cache] 文件缓存命中: {key[:30]}...")
                    # 加载到内存缓存
                    self._memory_cache[key] = entry
                    return entry["data"]
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"[Cache] 读取缓存失败: {e}")

        return None

    def set(self, key: str, data: dict):
        """设置缓存"""
        entry = {
            "timestamp": time.time(),
            "data": data,
        }
        # 写入内存缓存
        self._memory_cache[key] = entry
        # 写入文件缓存
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
            logger.info(f"[Cache] 缓存已保存: {key[:30]}...")
        except Exception as e:
            logger.warning(f"[Cache] 保存缓存失败: {e}")

    def _hash_key(self, key: str) -> str:
        """生成缓存文件名"""
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()

    def cleanup(self):
        """清理过期缓存"""
        now = time.time()
        cleaned = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if now - entry["timestamp"] > self.ttl:
                    cache_file.unlink()
                    cleaned += 1
            except Exception:
                cache_file.unlink()
                cleaned += 1
        if cleaned > 0:
            logger.info(f"[Cache] 已清理 {cleaned} 个过期缓存")


# 全局缓存实例
cache = CacheManager()
