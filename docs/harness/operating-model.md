# Operating Model

Harness work separates planning, implementation, evaluation, and handoff. A
single human or agent may perform multiple roles on small tasks, but the
artifacts should keep the roles distinct.

## Planner

The planner turns an input request into a scoped task brief.

Planner responsibilities:

- Identify the user or research outcome.
- Name in-scope and out-of-scope files, datasets, cases, and commands.
- Split complex work into contracts that can be evaluated independently.
- Record assumptions instead of hiding them in chat.
- Prefer repository-local context over external memory.

Planner output:

- `task-brief.md`
- An active plan folder for multi-step work
- Optional decision records for durable choices

## Generator

The generator implements the agreed contract.

Generator responsibilities:

- Read the task brief, contract, relevant code, and verification rules.
- Propose or update the sprint contract before changing code.
- Keep edits scoped to the contract.
- Preserve user changes and unrelated local work.
- Record commands run and files changed.

Generator output:

- Code, scripts, configs, or docs in the agreed scope
- Updated contract when scope changes
- Handoff notes for follow-up work

## Evaluator

The evaluator is skeptical and evidence-driven. Its job is to find gaps between
the contract and the implementation.

Evaluator responsibilities:

- Review the contract before implementation when possible.
- Run the cheapest checks that can catch real regressions.
- Escalate to GPU/data-heavy checks only when the task requires them.
- Record exact commands, data cases, outputs, and failures.
- Fail work for stubbed behavior, unverifiable claims, or missing evidence.

Evaluator output:

- `qa-report.md`
- Actionable findings with file paths and commands
- Residual risk notes when full verification is not available

## Handoff

Handoff preserves state across sessions.

Handoff responsibilities:

- Summarize changed behavior, not just changed files.
- Link the next agent to the first files to read.
- Record blockers, skipped checks, and data dependencies.
- Move completed task artifacts out of active plans.

Handoff output:

- `handoff.md`
- Updated `docs/plans/tech-debt-tracker.md` when debt remains

## Role Combination

For small changes, one agent can act as planner, generator, and evaluator. Keep
the artifacts explicit enough that another agent can audit the work without
reconstructing the session from chat.
