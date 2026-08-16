"""
Paper Agent - LLM 调用工具
统一重试、超时、降级逻辑
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import BaseMessage

logger = logging.getLogger("paper-agent")


def invoke_with_retry(
    llm,
    messages: list[BaseMessage],
    max_retries: int = 2,
    timeout: int = 30,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """带重试的 LLM 调用，失败时返回 fallback

    Args:
        llm: ChatOpenAI 实例
        messages: 消息列表
        max_retries: 最大重试次数
        timeout: 超时秒数（仅日志提示，实际由 ChatOpenAI.timeout 控制）
        fallback: 失败时的兜底返回值

    Returns:
        LLM 响应文本，失败返回 fallback 或 None
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            content = response.content.strip() if response.content else ""
            if content:
                return content
            logger.warning(f"[LLM] 第 {attempt + 1} 次调用返回空内容")
            last_error = "empty response"
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM] 第 {attempt + 1} 次调用失败: {e}")
            if attempt < max_retries:
                import time
                time.sleep(1 * (attempt + 1))  # 递增退避

    logger.error(f"[LLM] {max_retries + 1} 次调用均失败: {last_error}")
    return fallback


def invoke_json_with_retry(
    llm,
    messages: list[BaseMessage],
    max_retries: int = 2,
    fallback: Optional[dict] = None,
) -> Optional[dict]:
    """带重试的 JSON 输出 LLM 调用

    自动从 LLM 输出中提取 JSON，解析失败时重试。

    Returns:
        解析后的 dict，失败返回 fallback 或 None
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            content = response.content.strip() if response.content else ""
            if not content:
                last_error = "empty response"
                continue

            # 提取 JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            last_error = f"no JSON found in: {content[:100]}"
            logger.warning(f"[LLM] 第 {attempt + 1} 次未找到 JSON: {last_error}")
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"[LLM] 第 {attempt + 1} 次 JSON 解析失败: {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM] 第 {attempt + 1} 次调用失败: {e}")
            if attempt < max_retries:
                import time
                time.sleep(1 * (attempt + 1))

    logger.error(f"[LLM] JSON 调用 {max_retries + 1} 次均失败: {last_error}")
    return fallback
