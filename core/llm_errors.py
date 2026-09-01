"""Classify LLM failures and provide deterministic user-facing fallbacks."""

from __future__ import annotations

from typing import Any


LLM_QUOTA_EXHAUSTED = "LLM_QUOTA_EXHAUSTED"
LLM_RATE_LIMITED = "LLM_RATE_LIMITED"

_RATE_LIMIT_PATTERNS = (
    "rpm exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "requests per minute",
)
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "insufficient\\_quota",
    "allocated quota exceeded",
    "increase your quota limit",
    "quota exhausted",
    "quota_exceeded_error",
)


def classify_llm_error(error: object) -> str | None:
    """Return a stable code for provider quota/rate-limit errors."""
    normalized = str(error or "").strip().lower()
    if any(pattern in normalized for pattern in _RATE_LIMIT_PATTERNS):
        return LLM_RATE_LIMITED
    if any(pattern in normalized for pattern in _QUOTA_PATTERNS):
        return LLM_QUOTA_EXHAUSTED
    return None


def user_facing_llm_error(error: object) -> str | None:
    """Map either a provider error or a stable code to a local response."""
    value = str(error or "").strip()
    code = value if value in {LLM_QUOTA_EXHAUSTED, LLM_RATE_LIMITED} else classify_llm_error(value)
    if code == LLM_QUOTA_EXHAUSTED:
        return (
            "当前 LLM API 配额已用尽，系统已停止本轮后续模型调用。"
            "请补充配额或切换到可用模型后重试。"
        )
    if code == LLM_RATE_LIMITED:
        return (
            "当前 LLM API 请求频率额度已用尽，系统已停止本轮后续模型调用。"
            "请稍后重试。"
        )
    return None


def agent_failure_result(agent_name: str, error: Exception) -> dict[str, Any]:
    """Normalize an escaped agent exception without making another LLM call."""
    detail = str(error)
    code = classify_llm_error(detail)
    if code:
        return {
            "error": code,
            "error_detail": detail,
            "answer": user_facing_llm_error(code),
        }
    return {
        "error": f"{agent_name} 执行失败: {detail}",
        "error_detail": detail,
    }
