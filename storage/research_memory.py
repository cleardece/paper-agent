"""Structured, source-traceable long-term memory for a local researcher."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pymongo import ASCENDING, DESCENDING


logger = logging.getLogger("paper-agent")
PROFILE_FIELDS = {"learning_status", "research_directions", "projects", "preferences"}


class ResearchMemoryService:
    """Keep a profile snapshot plus an append-only audit trail of its changes."""

    def __init__(self, db: Any, llm: Any = None):
        self.profiles = db["research_profiles"]
        self.events = db["research_memory_events"]
        self.llm = llm
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.profiles.create_index([("user_id", ASCENDING)], unique=True)
        self.events.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        self.events.create_index([("source_session_id", ASCENDING)])

    @staticmethod
    def _default_profile(user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "learning_status": [],
            "research_directions": [],
            "projects": [],
            "preferences": [],
        }

    def get_profile(self, user_id: str) -> dict[str, Any]:
        profile = self.profiles.find_one({"user_id": user_id})
        if not profile:
            return self._default_profile(user_id)
        profile.pop("_id", None)
        for field in PROFILE_FIELDS:
            profile.setdefault(field, [])
        return profile

    def extract_updates(
        self,
        user_message: str,
        conversation_summary: str,
        session_focus: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract only user-profile updates; any model failure is non-fatal."""
        if self.llm is None:
            return self._extract_explicit_updates(user_message)

        prompt = f"""从用户本人这句话中提取长期研究档案的更新。

允许字段仅限：learning_status、research_directions、projects、preferences。
禁止把助手的回答、论文事实、推测或临时问题写进档案。若没有明确且有价值的用户信息，输出 []。
每项必须是 JSON 对象：{{"field": "允许字段", "value": ["值"], "mode": "append" 或 "set", "confidence": 0 到 1, "explicit": true 或 false}}。
当用户说“改为”“不是这样”“删除”时，使用 mode="set" 且 explicit=true。

会话摘要（仅帮助理解，不能作为新事实）：{conversation_summary[:800]}
会话焦点：{json.dumps(session_focus, ensure_ascii=False, default=str)}
用户原话：{user_message}
只输出 JSON 数组。"""
        try:
            response = self.llm.invoke([
                SystemMessage(content="你是严格、可审计的研究者档案提炼器。"),
                HumanMessage(content=prompt),
            ])
            content = getattr(response, "content", "")
            if not isinstance(content, str):
                return []
            match = re.search(r"\[[\s\S]*\]", content)
            parsed = json.loads(match.group()) if match else []
            return parsed if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning(f"[ResearchMemory] 提炼失败，跳过本轮更新: {exc}")
            return []

    @staticmethod
    def _extract_explicit_updates(message: str) -> list[dict[str, Any]]:
        """Offline fallback: recognize only unambiguous user corrections."""
        text = message.strip()
        direction = re.search(r"(?:研究方向)?(?:改为|是)\s*([^，。；;]+)", text)
        if direction and any(marker in text for marker in ("改为", "研究方向")):
            return [{
                "field": "research_directions",
                "value": [direction.group(1).strip()],
                "mode": "set",
                "confidence": 1.0,
                "explicit": True,
            }]
        return []

    @staticmethod
    def _normalized_values(value: Any) -> list[Any]:
        values = value if isinstance(value, list) else [value]
        return [item for item in values if isinstance(item, (str, dict)) and item]

    @staticmethod
    def _append_unique(existing: list[Any], incoming: list[Any]) -> list[Any]:
        result = list(existing)
        fingerprints = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in result}
        for item in incoming:
            fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if fingerprint not in fingerprints:
                result.append(item)
                fingerprints.add(fingerprint)
        return result

    def apply_updates(
        self,
        user_id: str,
        updates: list[dict[str, Any]],
        source_session_id: str,
        source_message_ids: list[str],
    ) -> dict[str, Any]:
        """Apply valid updates and retain an event for every material change."""
        profile = self.get_profile(user_id)
        now = datetime.now(timezone.utc)

        for update in updates:
            field = update.get("field")
            values = self._normalized_values(update.get("value"))
            if field not in PROFILE_FIELDS or not values:
                continue

            old_value = list(profile.get(field, []))
            new_value = values if update.get("mode") == "set" else self._append_unique(old_value, values)
            if new_value == old_value:
                continue

            profile[field] = new_value
            self.events.insert_one({
                "user_id": user_id,
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
                "source_session_id": source_session_id,
                "source_message_ids": list(source_message_ids),
                "confidence": float(update.get("confidence", 0.5)),
                "explicit": bool(update.get("explicit", False)),
                "status": "active",
                "updated_at": now,
            })

        profile["updated_at"] = now
        self.profiles.update_one(
            {"user_id": user_id},
            {"$set": profile, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return self.get_profile(user_id)
