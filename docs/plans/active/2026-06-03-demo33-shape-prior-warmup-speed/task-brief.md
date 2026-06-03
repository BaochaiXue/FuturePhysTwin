# Task Brief

## Task ID

`2026-06-03-demo33-shape-prior-warmup-speed`

## Request

Improve Demo 3.3 shape-prior warmup speed by optimizing flow and parallelism
without lowering quality parameters or skipping required shape-prior stages.

## Outcome

The single-view SAM3D shape-prior route remains complete, but the slow align and
sampling stages spend less wall time. Demo 3.3 still produces `final_data.pkl`
and a validated render-only shape prior.

## Scope

- In scope:
  - `data_process/align.py`
  - `data_process/utils/align_util.py`
  - `data_process_sam3d/shape_prior.py`
  - `data_process_sam3d/data_process_sample.py`
  - QQTT Demo 3.3 command wiring that can request data-only sampling output.
- Out of scope:
  - Reducing SAM3D inference steps, alignment pose counts, matching thresholds,
    image size, point targets, or tracker settings.
  - Replacing the original SAM3D backend.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/README.md`
- `data_process/align.py`
- `data_process/utils/align_util.py`
- `data_process_sam3d/data_process_sample.py`

## Data And Hardware Assumptions

- Required data: a Demo 3.3 captured case with shape prior artifacts.
- Required device: local RTX GPU for full route validation.
- Expected runtime: route-level checks can take minutes; unit/static checks are
  short.
- External services: none beyond already-installed checkpoints/packages.

## Acceptance Criteria

- Criterion 1: Align ray-registration uses equivalent constraints with less
  Python per-ray overhead.
- Criterion 2: Demo 3.3 can request data-only route visualization behavior
  while preserving SAM3D mesh export, sampling, and `final_data.pkl`.
- Criterion 3: Existing harness checks and focused route tests pass.

## Verification Plan

- Static checks: `python -m py_compile` for changed modules.
- Functional checks: run focused shape-prior/sample smoke on an existing Demo
  3.3 case.
- Full checks: `python scripts/check_harness_docs.py` and QQTT `check_all.py`
  where applicable.

## Risks

- Risk: vectorized ray casting changes duplicate constraint ordering.
- Mitigation: preserve first-seen ordering and only change batching, not
  thresholds or target selection rules.
