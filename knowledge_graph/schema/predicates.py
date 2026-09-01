"""Closed predicate vocabulary, stance values, and compatibility mapping."""

from __future__ import annotations

import re

PREDICATES = frozenset({
    "USES_METHOD", "USES_MODEL", "USES_DATASET", "USES_SOFTWARE", "SOLVES",
    "EVALUATED_BY", "IMPROVES", "OUTPERFORMS", "COMPARES_WITH", "EXTENDS",
    "BASED_ON", "UNKNOWN",
})
STANCES = frozenset({"support", "contradict"})
LEGACY_RELATIONS = frozenset({
    "proposes", "uses", "improves", "compares_with", "evaluates_on",
    "measures_with", "studies",
})

_EXACT_ALIASES = {
    "uses method": "USES_METHOD", "use method": "USES_METHOD",
    "uses model": "USES_MODEL", "use model": "USES_MODEL",
    "uses dataset": "USES_DATASET", "use dataset": "USES_DATASET",
    "evaluates on": "USES_DATASET", "evaluated on": "USES_DATASET",
    "uses software": "USES_SOFTWARE", "use software": "USES_SOFTWARE",
    "solves": "SOLVES", "solve": "SOLVES", "addresses": "SOLVES",
    "was employed to address": "SOLVES", "employed to solve": "SOLVES",
    "evaluated by": "EVALUATED_BY", "measured by": "EVALUATED_BY",
    "improves": "IMPROVES", "improve": "IMPROVES",
    "outperforms": "OUTPERFORMS", "outperform": "OUTPERFORMS",
    "compares with": "COMPARES_WITH", "compared with": "COMPARES_WITH",
    "extends": "EXTENDS", "extend": "EXTENDS", "proposes": "EXTENDS",
    "based on": "BASED_ON", "is based on": "BASED_ON",
}

_LEGACY_TO_CANONICAL = {
    "proposes": "EXTENDS", "uses": "USES_METHOD", "improves": "IMPROVES",
    "compares_with": "COMPARES_WITH", "evaluates_on": "USES_DATASET",
    "measures_with": "EVALUATED_BY", "studies": "SOLVES",
}

_TO_LEGACY = {
    "USES_METHOD": "uses", "USES_MODEL": "uses", "USES_DATASET": "uses",
    "USES_SOFTWARE": "uses", "SOLVES": "studies",
    "EVALUATED_BY": "measures_with", "IMPROVES": "improves",
    "OUTPERFORMS": "compares_with", "COMPARES_WITH": "compares_with",
    "EXTENDS": "proposes", "BASED_ON": "uses", "UNKNOWN": "studies",
}


def normalize_predicate(value: object) -> str:
    """Map a raw expression to the closed vocabulary or ``UNKNOWN``."""
    raw = str(value or "").strip()
    upper = raw.upper().replace("-", "_").replace(" ", "_")
    if upper in PREDICATES:
        return upper
    lower = re.sub(r"[_\s-]+", " ", raw.lower()).strip()
    if lower in _EXACT_ALIASES:
        return _EXACT_ALIASES[lower]
    legacy_key = lower.replace(" ", "_")
    if legacy_key in _LEGACY_TO_CANONICAL:
        return _LEGACY_TO_CANONICAL[legacy_key]
    return "UNKNOWN"


def predicate_for_legacy_relation(relation: object, object_type: str | None = None) -> str:
    relation_name = str(relation or "").strip().lower()
    if relation_name == "uses":
        return {
            "model": "USES_MODEL", "dataset": "USES_DATASET",
            "software": "USES_SOFTWARE",
        }.get(str(object_type or "").lower(), "USES_METHOD")
    return _LEGACY_TO_CANONICAL.get(relation_name, normalize_predicate(relation))


def to_legacy_relation(predicate: object) -> str:
    return _TO_LEGACY.get(normalize_predicate(predicate), "studies")
