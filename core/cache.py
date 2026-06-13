"""
Paper Agent - 缓存模块
会话级缓存：每个会话独立缓存，删除会话时连带删除
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("paper-agent")


class SessionCache:
    """会话级缓存管理器"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: dict[str, dict] = {}  # {session_id: {key: entry}}

    def _get_session_dir(self, session_id: str) -> Path:
        """获取会话缓存目录"""
        session_dir = self.cache_dir / session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir

    def get(self, session_id: str, key: str) -> dict | None:
        """获取会话级缓存"""
        if not session_id:
            return None

        cache_key = f"{session_id}:{key}"

        # 先查内存缓存
        if session_id in self._memory_cache and key in self._memory_cache[session_id]:
            entry = self._memory_cache[session_id][key]
            logger.info(f"[Cache] 内存缓存命中: {key[:30]}...")
            return entry["data"]

        # 再查文件缓存
        session_dir = self._get_session_dir(session_id)
        cache_file = session_dir / f"{self._hash_key(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                # 加载到内存缓存
                if session_id not in self._memory_cache:
                    self._memory_cache[session_id] = {}
                self._memory_cache[session_id][key] = entry
                logger.info(f"[Cache] 文件缓存命中: {key[:30]}...")
                return entry["data"]
            except Exception as e:
                logger.warning(f"[Cache] 读取缓存失败: {e}")

        return None

    def set(self, session_id: str, key: str, data: dict):
        """设置会话级缓存"""
        if not session_id:
            return

        entry = {
            "timestamp": time.time(),
            "data": data,
        }

        # 写入内存缓存
        if session_id not in self._memory_cache:
            self._memory_cache[session_id] = {}
        self._memory_cache[session_id][key] = entry

        # 写入文件缓存
        session_dir = self._get_session_dir(session_id)
        cache_file = session_dir / f"{self._hash_key(key)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
            logger.info(f"[Cache] 缓存已保存: {key[:30]}...")
        except Exception as e:
            logger.warning(f"[Cache] 保存缓存失败: {e}")

    def delete_session(self, session_id: str):
        """删除会话及其所有缓存"""
        # 删除内存缓存
        self._memory_cache.pop(session_id, None)

        # 删除文件缓存
        session_dir = self.cache_dir / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            logger.info(f"[Cache] 已删除会话缓存: {session_id}")

    def delete_all(self):
        """删除所有缓存"""
        self._memory_cache.clear()
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        logger.info("[Cache] 已删除所有缓存")

    def _hash_key(self, key: str) -> str:
        """生成缓存文件名"""
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()


# 全局缓存实例
cache = SessionCache()
