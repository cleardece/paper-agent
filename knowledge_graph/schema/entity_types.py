"""Closed entity-type vocabulary and deterministic normalization."""

from __future__ import annotations

import re

ENTITY_TYPES = frozenset({
    "method", "model", "algorithm", "dataset", "problem", "task",
    "equation", "software", "metric", "domain",
})

_ALIASES = {
    "method": "method", "methods": "method", "physics method": "method",
    "technique": "method", "approach": "method",
    "model": "model", "models": "model", "architecture": "model",
    "algorithm": "algorithm", "algorithms": "algorithm",
    "dataset": "dataset", "datasets": "dataset", "data set": "dataset",
    "corpus": "dataset", "benchmark": "dataset",
    "problem": "problem", "problems": "problem",
    "task": "task", "tasks": "task",
    "equation": "equation", "equations": "equation", "pde": "equation",
    "software": "software", "tool": "software", "framework": "software",
    "library": "software",
    "metric": "metric", "metrics": "metric", "measure": "metric",
    "domain": "domain", "field": "domain",
}


def normalize_entity_type(value: object) -> str | None:
    """Return a schema value or ``None``; never invent a new type."""
    normalized = re.sub(r"[_\s-]+", " ", str(value or "").strip().lower())
    return _ALIASES.get(normalized)
