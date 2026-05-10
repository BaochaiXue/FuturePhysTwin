#!/usr/bin/env python3
"""Validate the harness-engineering documentation scaffold."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "AGENTS.md",
    "HARNESS.md",
    "docs/harness/README.md",
    "docs/harness/context-map.md",
    "docs/harness/operating-model.md",
    "docs/harness/workflow.md",
    "docs/harness/verification.md",
    "docs/harness/invariants.md",
    "docs/harness/prompts/planner.md",
    "docs/harness/prompts/generator.md",
    "docs/harness/prompts/evaluator.md",
    "docs/harness/prompts/handoff.md",
    "docs/harness/templates/task-brief.md",
    "docs/harness/templates/sprint-contract.md",
    "docs/harness/templates/qa-report.md",
    "docs/harness/templates/handoff.md",
    "docs/harness/templates/decision-record.md",
    "docs/harness/templates/agent-run-log.md",
    "docs/plans/active",
    "docs/plans/completed",
    "docs/plans/tech-debt-tracker.md",
]

REQUIRED_REFERENCES = {
    "AGENTS.md": ["HARNESS.md", "docs/harness/README.md"],
    "HARNESS.md": ["docs/harness/README.md", "docs/plans/active/"],
    "docs/harness/README.md": [
        "context-map.md",
        "operating-model.md",
        "workflow.md",
        "verification.md",
        "invariants.md",
        "prompts/",
        "templates/",
    ],
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_required_paths(errors: list[str]) -> None:
    for item in REQUIRED_PATHS:
        path = ROOT / item
        if not path.exists():
            errors.append(f"missing required path: {item}")


def check_markdown_headings(errors: list[str]) -> None:
    for path in (ROOT / "docs/harness").rglob("*.md"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            errors.append(f"empty markdown file: {rel(path)}")
            continue
        if not text.startswith("# "):
            errors.append(f"markdown file must start with H1: {rel(path)}")

    for item in ["HARNESS.md", "docs/plans/tech-debt-tracker.md"]:
        path = ROOT / item
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if not text.startswith("# "):
                errors.append(f"markdown file must start with H1: {item}")


def check_references(errors: list[str]) -> None:
    for source, refs in REQUIRED_REFERENCES.items():
        path = ROOT / source
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for ref in refs:
            if ref not in text:
                errors.append(f"{source} must reference {ref}")


def main() -> int:
    errors: list[str] = []
    check_required_paths(errors)
    check_markdown_headings(errors)
    check_references(errors)

    if errors:
        for error in errors:
            print(f"harness-docs: {error}", file=sys.stderr)
        return 1

    print("harness-docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
