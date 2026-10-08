"""Exercise the real child entry point without a network or database."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CHILD_STUB = r'''
import runpy
import sys
import types

scenario = sys.argv[1]

class Extractor:
    def __init__(self, llm):
        pass

    def extract_batch(self, paper, batch):
        if scenario == "error":
            raise ValueError("model unavailable")
        if scenario == "timeout":
            raise type("APITimeoutError", (Exception,), {})("Request timed out.")
        if scenario == "connection":
            raise type("APIConnectionError", (Exception,), {})("Connection error.")
        if scenario == "protocol_error":
            error = ValueError("incomplete model response")
            error.error_kind = "llm_output_truncated"
            error.diagnostics = {"finish_reason": "length"}
            raise error
        if scenario == "serialization_error":
            return {"candidates": [{"evidence": "prefix"}, {"bad": object()}]}
        if scenario == "noise":
            print("SDK debug output")
        return {"candidates": [{"evidence": paper["title"]}]}

    def validate_batch(self, paper, batch, candidates):
        return {"relations": candidates}

def get_llm():
    return object()

def resolve(llm, items):
    return {"decisions": items}

for name, attributes in (
    ("agents.research_graph_extractor", {"ResearchGraphExtractor": Extractor}),
    ("config", {"get_graph_llm": get_llm}),
    ("knowledge_graph.entity_resolution.llm_resolver", {"resolve_entity_batch": resolve}),
    ("knowledge_graph.fact_resolution.llm_resolver", {"resolve_fact_batch": resolve}),
):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module

runpy.run_module("tools.research_graph_process", run_name="__main__")
'''


def run_child(scenario, payload, encoding="utf-8", escaped_input=False):
    return subprocess.run(
        [sys.executable, "-c", CHILD_STUB, scenario],
        input=json.dumps(payload, ensure_ascii=escaped_input).encode("utf-8"),
        capture_output=True,
        timeout=10,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONIOENCODING": encoding},
    )


@pytest.mark.parametrize("escaped_input", [False, True])
def test_child_uses_utf8_even_with_legacy_windows_stream_encoding(escaped_input):
    evidence = "科学公式 \U0001d6fc and \U0001f9ea"
    result = run_child(
        "success", {"paper": {"title": evidence}}, encoding="gbk",
        escaped_input=escaped_input,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["ok"] is True
    assert data["candidates"] == [{"evidence": evidence}]


def test_serialization_error_returns_one_complete_error_document():
    result = run_child("serialization_error", {"paper": {"title": "paper"}})

    assert result.returncode == 1
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["ok"] is False
    assert "not JSON serializable" in data["error"]
    assert "TypeError" in data["traceback"]


def test_sdk_stdout_is_redirected_to_stderr():
    result = run_child("noise", {"paper": {"title": "paper"}})

    assert result.returncode == 0
    assert json.loads(result.stdout.decode("utf-8"))["ok"] is True
    assert b"SDK debug output" in result.stderr


def test_model_error_returns_complete_error_envelope():
    result = run_child("error", {"paper": {"title": "paper"}})

    assert result.returncode == 1
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["ok"] is False
    assert data["error"] == "model unavailable"


@pytest.mark.parametrize("scenario, error_kind", [
    ("timeout", "llm_request_timeout"),
    ("connection", "llm_connection_error"),
    ("protocol_error", "llm_output_truncated"),
])
def test_error_kind_and_safe_diagnostics_survive_transport(scenario, error_kind):
    result = run_child(scenario, {"paper": {"title": "paper"}})

    assert result.returncode == 1
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["error_kind"] == error_kind
    assert data["mode"] == "extract"
    assert data["elapsed_seconds"] >= 0
    if scenario == "protocol_error":
        assert data["diagnostics"] == {"finish_reason": "length"}


@pytest.mark.parametrize("mode, key", [
    ("extract", "candidates"),
    ("validate", "relations"),
    ("resolve_entities", "decisions"),
    ("resolve_facts", "decisions"),
])
def test_all_four_modes_keep_success_envelope(mode, key):
    result = run_child("success", {"mode": mode, "paper": {"title": "paper"}})

    assert result.returncode == 0
    data = json.loads(result.stdout.decode("utf-8"))
    assert data["ok"] is True
    assert key in data
