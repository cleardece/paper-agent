"""Run labelled research questions through the existing workflow and report metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evidence import validate_answer_evidence
from evaluation.metrics import summarize_runs
from main import create_initial_state, init_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, citation evidence, abstention, and latency."
    )
    parser.add_argument("--cases", required=True, type=Path, help="Labelled JSON case file")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation case file does not exist: {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation case file must contain a JSON array")
    required = {"query", "expected_papers", "should_abstain"}
    for index, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case {index} is missing: {', '.join(sorted(missing))}")
    return cases


def run_cases(cases: list[dict]) -> list[dict]:
    workflow = init_components()
    runs: list[dict] = []
    for case in cases:
        state = create_initial_state(case["query"])
        state["max_iterations"] = 2
        started_at = time.perf_counter()
        result = asyncio.run(workflow.ainvoke(state))
        latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
        chunks = result.get("retrieved_chunks", [])
        answer_for_check = result.get("analysis") or result.get("answer") or ""
        evidence = result.get("evidence_report") or validate_answer_evidence(
            answer_for_check, chunks
        )
        runs.append(
            {
                "case_id": case.get("id", case["query"][:40]),
                "expected_papers": case["expected_papers"],
                "retrieved_papers": list(
                    dict.fromkeys(chunk.get("paper_arxiv_id", "") for chunk in chunks)
                ),
                "should_abstain": case["should_abstain"],
                "evidence_status": evidence["status"],
                "latency_ms": latency_ms,
            }
        )
    return runs


def main() -> None:
    args = parse_args()
    runs = run_cases(load_cases(args.cases))
    print(json.dumps({"summary": summarize_runs(runs), "runs": runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
