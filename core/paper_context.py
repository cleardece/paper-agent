"""Deterministic paper identity resolution for a conversation turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_utils import invoke_json_with_retry


ARXIV_ID_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/|\s*:\s*)?)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
COMPARE_MARKERS = ("对比", "比较", "区别", "compare", "versus", " vs ")
SWITCH_MARKERS = ("再看看", "再看", "切换", "换一篇", "换成", "改看", "另一篇")
SEARCH_MARKERS = (
    "搜索", "查找", "找论文", "找几篇", "类似论文", "相似论文", "相关论文",
    "检索文献", "search", "find papers", "look for papers",
)
SIMILAR_MARKERS = ("类似", "相似", "similar")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())


def is_explicit_search_request(query: str) -> bool:
    lowered = f" {query.casefold()} "
    if any(marker in lowered for marker in SEARCH_MARKERS):
        return True
    return bool(
        re.search(
            r"(?:帮我)?(?:找|搜|查)[^，。；;!?！？]{0,120}(?:论文|文献)",
            query,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:find|search(?:\s+for)?|look\s+for)\b.{0,120}\bpapers?\b",
            query,
            re.IGNORECASE,
        )
    )


def is_similar_paper_search(query: str) -> bool:
    lowered = query.casefold()
    return is_explicit_search_request(query) and any(
        marker in lowered for marker in SIMILAR_MARKERS
    )


@dataclass
class PaperFocusState:
    primary_paper_id: str | None = None
    active_paper_ids: list[str] = field(default_factory=list)
    source: str | None = None
    confidence: float | None = None
    last_resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        self.active_paper_ids = _unique(self.active_paper_ids)
        if self.primary_paper_id:
            self.active_paper_ids = _unique(
                [self.primary_paper_id, *self.active_paper_ids]
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_paper_id": self.primary_paper_id,
            "active_paper_ids": list(self.active_paper_ids),
            "source": self.source,
            "confidence": self.confidence,
            "last_resolved_at": self.last_resolved_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "PaperFocusState":
        value = value or {}
        timestamp = value.get("last_resolved_at")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = None
        active_ids = list(value.get("active_paper_ids") or [])
        primary_id = value.get("primary_paper_id") or (
            active_ids[0] if active_ids else None
        )
        return cls(
            primary_paper_id=primary_id,
            active_paper_ids=active_ids,
            source=value.get("source"),
            confidence=value.get("confidence"),
            last_resolved_at=timestamp,
        )


@dataclass(frozen=True)
class PaperContext:
    primary_paper_id: str | None
    paper_ids: list[str]
    status: str
    source: str
    confidence: float
    inherited: bool
    switched_from_paper_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_paper_id": self.primary_paper_id,
            "paper_ids": list(self.paper_ids),
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "inherited": self.inherited,
            "switched_from_paper_id": self.switched_from_paper_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "PaperContext":
        value = value or {}
        return cls(
            primary_paper_id=value.get("primary_paper_id"),
            paper_ids=list(value.get("paper_ids") or []),
            status=value.get("status", "unresolved"),
            source=value.get("source", "none"),
            confidence=float(value.get("confidence") or 0.0),
            inherited=bool(value.get("inherited", False)),
            switched_from_paper_id=value.get("switched_from_paper_id"),
        )


class PaperContextResolver:
    """Resolve paper identity without granting external search capability."""

    def __init__(self, paper_repository, llm=None):
        self.repository = paper_repository
        self.llm = llm

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        focus_data = state.get("paper_focus") or {
            "primary_paper_id": state.get("primary_paper_id"),
            "active_paper_ids": state.get("active_paper_ids") or [],
        }
        context = self.resolve(
            query=state["user_query"],
            explicit_target_paper_id=state.get("target_paper_id"),
            paper_focus=PaperFocusState.from_dict(focus_data),
            recent_paper_contexts=state.get("recent_paper_contexts") or [],
        )
        return {
            "paper_context": context.to_dict(),
            "primary_paper_id": context.primary_paper_id,
        }

    def resolve(
        self,
        query: str,
        explicit_target_paper_id: str | None,
        paper_focus: PaperFocusState,
        recent_paper_contexts: list[dict[str, Any]],
    ) -> PaperContext:
        current_id = paper_focus.primary_paper_id
        if explicit_target_paper_id:
            return self._resolved(
                explicit_target_paper_id,
                [explicit_target_paper_id],
                "explicit_target",
                1.0,
                current_id,
            )

        papers = self._list_papers()
        arxiv_match = ARXIV_ID_RE.search(query)
        if arxiv_match:
            paper_id = arxiv_match.group(1)
            if self._paper_exists(paper_id, papers):
                return self._resolved(
                    paper_id, [paper_id], "arxiv_id", 1.0, current_id
                )

        doi_match = DOI_RE.search(query)
        if doi_match:
            doi = doi_match.group(0).rstrip(".,;，。；")
            matches = [
                item for item in papers
                if str(item.get("doi", "")).casefold() == doi.casefold()
            ]
            if len(matches) == 1:
                paper_id = matches[0]["arxiv_id"]
                return self._resolved(
                    paper_id, [paper_id], "doi", 1.0, current_id
                )

        title_matches = self._title_matches(query, papers)
        if title_matches:
            if self._is_compare(query) and current_id:
                ids = _unique([current_id, *title_matches])
                return PaperContext(
                    primary_paper_id=current_id,
                    paper_ids=ids,
                    status="resolved",
                    source="title_match",
                    confidence=0.97,
                    inherited=True,
                )
            if len(title_matches) == 1:
                paper_id = title_matches[0]
                return self._resolved(
                    paper_id, [paper_id], "title_match", 0.97, current_id
                )
            selected = self._resolve_ambiguous(query, title_matches, papers)
            if selected:
                return self._resolved(
                    selected, [selected], "title_match", 0.8, current_id
                )
            return PaperContext(
                primary_paper_id=None,
                paper_ids=title_matches,
                status="ambiguous",
                source="title_match",
                confidence=0.5,
                inherited=False,
            )

        if self._is_switch(query):
            return self._unresolved()

        if is_explicit_search_request(query) and not is_similar_paper_search(query):
            return self._unresolved()

        if current_id:
            return PaperContext(
                primary_paper_id=current_id,
                paper_ids=_unique([current_id, *paper_focus.active_paper_ids]),
                status="resolved",
                source="session_focus",
                confidence=0.98,
                inherited=True,
            )

        for metadata in reversed(recent_paper_contexts):
            historical = PaperContext.from_dict(metadata)
            if historical.primary_paper_id and historical.status in {
                "resolved", "switch"
            }:
                return PaperContext(
                    primary_paper_id=historical.primary_paper_id,
                    paper_ids=_unique(
                        [historical.primary_paper_id, *historical.paper_ids]
                    ),
                    status="resolved",
                    source="history_resolution",
                    confidence=min(historical.confidence, 0.9),
                    inherited=True,
                )

        return self._unresolved()

    @staticmethod
    def _resolved(
        paper_id: str,
        paper_ids: list[str],
        source: str,
        confidence: float,
        current_id: str | None,
    ) -> PaperContext:
        switched = current_id if current_id and current_id != paper_id else None
        return PaperContext(
            primary_paper_id=paper_id,
            paper_ids=_unique(paper_ids),
            status="switch" if switched else "resolved",
            source=source,
            confidence=confidence,
            inherited=False,
            switched_from_paper_id=switched,
        )

    @staticmethod
    def _unresolved() -> PaperContext:
        return PaperContext(
            primary_paper_id=None,
            paper_ids=[],
            status="unresolved",
            source="none",
            confidence=0.0,
            inherited=False,
        )

    def _list_papers(self) -> list[dict[str, Any]]:
        try:
            return list(self.repository.list_papers(
                limit=200,
                projection={
                    "arxiv_id": 1,
                    "title": 1,
                    "doi": 1,
                    "abstract": 1,
                },
            ))
        except Exception:
            return []

    @staticmethod
    def _paper_exists(paper_id: str, papers: list[dict[str, Any]]) -> bool:
        canonical = paper_id.casefold()
        return any(
            str(item.get("arxiv_id", "")).casefold() == canonical
            for item in papers
        )

    @staticmethod
    def _title_matches(query: str, papers: list[dict[str, Any]]) -> list[str]:
        normalized_query = _normalize_title(query)
        matches = []
        for item in papers:
            paper_id = str(item.get("arxiv_id", ""))
            title = _normalize_title(str(item.get("title", "")))
            if paper_id and len(title) >= 6 and title in normalized_query:
                matches.append(paper_id)
        return _unique(matches)

    @staticmethod
    def _is_compare(query: str) -> bool:
        lowered = f" {query.casefold()} "
        return any(marker in lowered for marker in COMPARE_MARKERS)

    @staticmethod
    def _is_switch(query: str) -> bool:
        lowered = query.casefold()
        return any(marker in lowered for marker in SWITCH_MARKERS)

    def _resolve_ambiguous(
        self,
        query: str,
        candidate_ids: list[str],
        papers: list[dict[str, Any]],
    ) -> str | None:
        if self.llm is None:
            return None
        by_id = {str(item.get("arxiv_id")): item for item in papers}
        candidates = [
            {
                "paper_id": paper_id,
                "title": by_id.get(paper_id, {}).get("title", ""),
            }
            for paper_id in candidate_ids
        ]
        decision = invoke_json_with_retry(
            self.llm,
            [
                SystemMessage(content=(
                    "Choose the one paper_id best matching the user text. "
                    "You may only return an ID from candidates, or null. "
                    "Return JSON: {\"paper_id\": string|null}."
                )),
                HumanMessage(content=(
                    f"User text: {query}\nCandidates: "
                    f"{json.dumps(candidates, ensure_ascii=False)}"
                )),
            ],
            max_retries=1,
        )
        selected = decision.get("paper_id") if decision else None
        return selected if selected in candidate_ids else None
