# Sprint Contract

## Contract ID

`harness-bootstrap`

## Goal

Install the initial harness engineering scaffold without changing PhysTwin
runtime behavior.

## Agreed Scope

Files expected to change:

- `AGENTS.md`
- `HARNESS.md`
- `docs/harness/**`
- `docs/plans/**`
- `scripts/check_harness_docs.py`
- `.github/workflows/harness-docs.yml`

Files expected to read:

- `AGENTS.md`
- `README.md`
- `implement_plan.md`

Out of scope:

- Python training, reconstruction, simulation, rendering, and data processing
  source changes.
- Generated datasets or experiment outputs.

## Behavior To Deliver

- Agents can find a clear harness entry point from `AGENTS.md`.
- Future non-trivial work has templates for task brief, contract, QA, handoff,
  decisions, and run logs, plus reusable role prompts.
- Harness docs can be checked locally and in CI.

## Acceptance Criteria

- Must keep existing PhysTwin subtree instructions intact.
- Must add a standalone-checkout fallback for missing parent docs.
- Must include PhysTwin-specific verification guidance.
- Must not introduce bridge-specific conclusions.

## Evaluator Probes

The evaluator should explicitly check:

- `python scripts/check_harness_docs.py` passes.
- `python -m py_compile scripts/check_harness_docs.py` passes.
- `AGENTS.md` references `HARNESS.md` and `docs/harness/README.md`.
- The CI workflow runs only for harness-relevant paths.

## Verification Commands

```bash
python scripts/check_harness_docs.py
python -m py_compile scripts/check_harness_docs.py
```

## Contract Changes

None.
