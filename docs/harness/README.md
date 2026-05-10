# Harness Engineering

Harness engineering is the management system for this repository. It turns
agent work into a repeatable loop: define intent, agree on testable contracts,
implement in scoped passes, evaluate with evidence, and preserve handoff state.

This directory is the map. It should stay short enough to fit into agent
context, with deeper details split into focused files.

## Files

- `context-map.md`: repository topology and where agents should look first.
- `operating-model.md`: planner, generator, evaluator, and handoff roles.
- `workflow.md`: task lifecycle from prompt to completed plan.
- `verification.md`: PhysTwin-specific verification matrix.
- `invariants.md`: repository rules that should remain stable across tasks.
- `prompts/`: reusable role prompts for planner, generator, evaluator, and
  handoff agents.
- `templates/`: copyable artifacts for active work.

## Active Work

Active harness work lives under `docs/plans/active/<task-id>/`. Each task folder
should contain, at minimum:

- `task-brief.md`
- `sprint-contract.md` for implementation work
- `qa-report.md` before completion
- `handoff.md` when work spans sessions or leaves follow-ups

Completed work moves to `docs/plans/completed/<task-id>/`.

## Minimum Bar

Every non-trivial task must answer:

- What user or research outcome is being changed?
- Which files and data paths are in scope?
- Which commands or manual checks prove the result?
- What risks remain?
- What should the next agent read first?

## Agent Readability

Prefer durable, local, versioned context over long prompts. If an agent needs to
know something twice, write it down here or in a task plan. If a rule is
mechanically checkable, prefer a script or test over prose.
