"""Deterministic entity-name normalization."""

from __future__ import annotations

import re
import unicodedata


def _singularize_token(token: str) -> str:
    if token.endswith(("ics", "ss", "us", "is")):
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[‐‑‒–—−_/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [_singularize_token(token) for token in text.split()]
    return " ".join(tokens)


def acronym(value: object) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    tokens = normalized.split()
    compact = "".join(tokens)
    if len(tokens) == 1 and len(compact) <= 12:
        return compact
    return "".join(token[0] for token in tokens if token)


def blocking_keys(value: object) -> list[str]:
    normalized = normalize_name(value)
    if not normalized:
        return []
    tokens = normalized.split()
    values = {f"name:{normalized}", f"acro:{acronym(value)}"}
    if tokens:
        values.add(f"first:{tokens[0]}")
        values.add(f"prefix:{normalized[:8]}")
    return sorted(key for key in values if not key.endswith(":"))
