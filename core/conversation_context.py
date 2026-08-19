"""Build bounded, non-evidentiary context for a research conversation."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


SUMMARY_REFRESH_MESSAGE_COUNT = 10


def _message_value(message: Any, key: str, default: str = "") -> str:
    if isinstance(message, dict):
        return str(message.get(key, default))
    return str(getattr(message, key, default))


def build_conversation_context(
    summary: str,
    messages: list[Any],
    recent_message_limit: int = 10,
) -> str:
    """Combine a prior summary and a bounded tail of raw dialogue.

    The result helps resolve conversational intent only. It is never supplied as
    retrieved evidence or a citation source.
    """
    parts: list[str] = []
    normalized_summary = " ".join((summary or "").split())
    if normalized_summary:
        parts.append(f"早期对话摘要: {normalized_summary}")

    for message in messages[-recent_message_limit:]:
        role = _message_value(message, "role")
        content = " ".join(_message_value(message, "content").split())[:500]
        if not content:
            continue
        role_label = "用户" if role == "user" else "助手"
        parts.append(f"{role_label}: {content}")

    return "\n".join(parts)


def needs_summary_refresh(message_count: int, summary_through_message_count: int) -> bool:
    """Refresh only after a full block of previously unsummarized messages."""
    return message_count - summary_through_message_count >= SUMMARY_REFRESH_MESSAGE_COUNT


def refresh_summary(llm: Any, previous_summary: str, messages: list[Any]) -> str:
    """Best-effort conversation-state summary; failures preserve the old value."""
    if llm is None:
        return previous_summary

    context = build_conversation_context(previous_summary, messages, recent_message_limit=30)
    prompt = (
        "你负责压缩研究型对话的工作记忆。只总结用户已经明确表达的："
        "当前论文/章节、研究任务、未解决问题、研究进展和更正。"
        "不要把助手的推测当事实；不要声称论文事实；不要添加引用。"
        "输出一段简洁中文摘要，不超过 800 字。\n\n"
        f"对话：\n{context}"
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content="你是谨慎的研究对话记忆压缩器。"),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()[:800]
    except Exception:
        pass
    return previous_summary
