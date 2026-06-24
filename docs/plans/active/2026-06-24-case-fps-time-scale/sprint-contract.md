# Sprint Contract

## Contract ID

`case-fps-time-scale`

## Goal

Make real-case CMA optimization, first-order training, and inference derive
their per-observation simulation horizon from case metadata FPS.

## Agreed Scope

Files expected to change:

- `qqtt/utils/config.py`
- `optimize_cma.py`
- `train_warp.py`
- `inference_warp.py`
- `tests/test_case_fps_time_scale.py`
- `docs/plans/active/2026-06-24-case-fps-time-scale/qa-report.md`

Files expected to read:

- `configs/real.yaml`
- `qqtt/engine/trainer_warp.py`
- `qqtt/engine/cma_optimize_warp.py`
- `qqtt/model/diff_simulator/spring_mass_warp.py`

Out of scope:

- Full GPU optimization or training reruns.
- Demo v4 chunk writer changes.
- Reparameterizing material optimization for changed frame horizons.

## Behavior To Deliver

- `cfg.apply_case_timing_from_metadata(metadata)` reads `metadata["fps"]` when
  present, falls back to `cfg.FPS` when absent, validates positive FPS, and
  sets `FPS`, `num_substeps`, and `dt` so `dt * num_substeps == 1 / FPS`.
- `optimize_cma.py`, `train_warp.py`, and `inference_warp.py` invoke the helper
  after loading metadata and before constructing optimizer/trainer objects.

## Acceptance Criteria

- Must preserve a base integration step no larger than the configured `cfg.dt`.
- Must preserve current 30 FPS effective behavior for metadata without `fps`.
- Must fail early on zero or negative `fps`.
- Must not change data loading schemas or generated case files.

## Evaluator Probes

The evaluator should explicitly check:

- Probe: 5 FPS metadata yields 4000 substeps and 0.2 seconds per frame.
- Probe: the three entrypoints contain the shared helper call and no duplicate
  hand-rolled timing math.
- Probe: tests fail before implementation and pass after implementation.

## Verification Commands

```bash
pytest tests/test_case_fps_time_scale.py -q
python -m py_compile optimize_cma.py train_warp.py inference_warp.py qqtt/utils/config.py
python scripts/check_harness_docs.py
```

## Contract Changes

- 2026-06-24: Initial contract from user report that Demo v4 5 FPS data was
  being optimized as 30 FPS.
