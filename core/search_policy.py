"""External paper-search admission and query validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.paper_context import ARXIV_ID_RE, is_similar_paper_search


class ExternalSearchNotAllowed(RuntimeError):
    pass


class InvalidSearchRequest(ValueError):
    pass


@dataclass(frozen=True)
class SearchRequest:
    mode: str
    value: str

    def validate(self) -> None:
        if self.mode not in {"arxiv_id", "title", "keywords"}:
            raise InvalidSearchRequest(f"unsupported search mode: {self.mode}")
        if not self.value.strip():
            raise InvalidSearchRequest("search value is empty")
        if len(self.value) > 200:
            raise InvalidSearchRequest("search value is too long")

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "value": self.value}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchRequest":
        request = cls(
            mode=str(value.get("mode", "")),
            value=str(value.get("value", "")).strip(),
        )
        request.validate()
        return request


class SearchQueryBuilder:
    _quoted = re.compile(r"[\"“]([^\"”]{4,200})[\"”]")
    _analysis_tail = re.compile(
        r"[，,；;]?\s*(?:并|然后|再)?\s*"
        r"(?:分析|回答|解释|总结|评估|对比|比较)(?:以下|下面)?[\s\S]*$",
        re.IGNORECASE,
    )
    _latin_token = re.compile(
        r"[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*"
    )
    _latin_stopwords = {
        "arxiv", "find", "help", "look", "paper", "papers", "please",
        "search", "the", "for", "about", "and",
    }

    def build(
        self,
        query: str,
        fallback_title: str | None = None,
    ) -> SearchRequest:
        arxiv_match = ARXIV_ID_RE.search(query)
        if arxiv_match:
            return self._validated("arxiv_id", arxiv_match.group(1))

        quoted = self._quoted.search(query)
        if quoted:
            return self._validated("title", " ".join(quoted.group(1).split()))

        if is_similar_paper_search(query) and fallback_title:
            return self._validated(
                "keywords", " ".join(fallback_title.split())
            )

        discovery_part = self._analysis_tail.sub("", query).strip()
        latin_tokens = [
            token for token in self._latin_token.findall(discovery_part)
            if token.casefold() not in self._latin_stopwords
        ]
        if latin_tokens:
            return self._validated("keywords", " ".join(latin_tokens[:12]))

        chinese = re.sub(
            r"帮我|请|搜索|查找|找几篇|找一些|找|检索|相关|类似|相似|论文|文献|的",
            " ",
            discovery_part,
        )
        chinese = " ".join(chinese.split()).strip(" ：:，,。")
        if not chinese and fallback_title:
            chinese = " ".join(fallback_title.split())
        return self._validated("keywords", chinese)

    @staticmethod
    def _validated(mode: str, value: str) -> SearchRequest:
        request = SearchRequest(mode=mode, value=value.strip())
        request.validate()
        return request


class SearchAdmissionGate:
    """The single executable boundary in front of external paper search."""

    @staticmethod
    def ensure_allowed(turn_context: dict[str, Any]) -> None:
        if (
            turn_context.get("intent") != "search"
            or not turn_context.get("allow_external_search")
        ):
            raise ExternalSearchNotAllowed(
                "external search requires an admitted SEARCH turn"
            )

    def search(
        self,
        arxiv,
        turn_context: dict[str, Any],
        request: SearchRequest | dict[str, Any],
        max_results: int,
    ):
        self.ensure_allowed(turn_context)
        if isinstance(request, dict):
            request = SearchRequest.from_dict(request)
        request.validate()
        return arxiv.search(request.value, max_results=max_results)


_SEARCH_ADMISSION_GATE = SearchAdmissionGate()


def guarded_arxiv_search(
    arxiv,
    turn_context: dict[str, Any],
    request: SearchRequest | dict[str, Any],
    max_results: int,
):
    return _SEARCH_ADMISSION_GATE.search(
        arxiv, turn_context, request, max_results
    )


def ensure_external_search_allowed(turn_context: dict[str, Any]) -> None:
    _SEARCH_ADMISSION_GATE.ensure_allowed(turn_context)
