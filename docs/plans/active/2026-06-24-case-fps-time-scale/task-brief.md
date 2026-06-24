# Task Brief

## Task ID

`2026-06-24-case-fps-time-scale`

## Request

Fix real-case optimization, training, and inference so each observed frame
transition uses the FPS stored in the case `metadata.json` instead of always
using the default 30 FPS simulation horizon from `configs/real.yaml`.

## Outcome

Demo v4 cases with `metadata.json` containing `"fps": 5` run one simulation
frame over 0.2 seconds, not 0.03335 seconds. Existing 30 FPS cases keep the
same effective `dt`/`num_substeps` behavior.

## Scope

- In scope:
  - Shared config helper for case metadata FPS.
  - `optimize_cma.py`, `train_warp.py`, and `inference_warp.py`.
  - Focused tests for 5 FPS, fallback FPS, invalid FPS, and entrypoint wiring.
  - Harness QA evidence.
- Out of scope:
  - Demo v4 data generation changes.
  - Simulator physics model changes other than the per-frame time horizon.
  - Re-running expensive CMA/train/inference jobs.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/README.md`
- `configs/real.yaml`
- `qqtt/utils/config.py`
- `optimize_cma.py`
- `train_warp.py`
- `inference_warp.py`
- `qqtt/model/diff_simulator/spring_mass_warp.py`

## Data And Hardware Assumptions

- Required data: no full real case required for focused unit tests.
- Required device: CPU is enough for config and entrypoint wiring tests.
- Expected runtime: focused tests should complete in seconds.
- External services: none.

## Acceptance Criteria

- 5 FPS metadata sets `cfg.FPS = 5.0`, `cfg.num_substeps = 4000`, and
  `cfg.dt = 5e-5` when the base step is `5e-5`.
- Missing metadata FPS preserves the configured FPS frame horizon and does not
  accidentally change 30 FPS behavior.
- Non-positive metadata FPS raises a clear `ValueError`.
- All three main entrypoints call the shared FPS helper immediately after
  loading `metadata.json`.

## Verification Plan

- Static checks: `python -m py_compile optimize_cma.py train_warp.py inference_warp.py qqtt/utils/config.py`
- Functional checks: `pytest tests/test_case_fps_time_scale.py -q`
- Harness checks: `python scripts/check_harness_docs.py`

## Risks

- Risk: increasing 5 FPS cases from 667 to 4000 substeps increases runtime.
- Mitigation: preserve base `dt` and only expand substeps to match the true
  frame interval; this is the intended physical correction.
