# QA Report

## Task ID

`2026-06-24-case-fps-time-scale`

## Summary

Focused TDD, static compilation, whitespace, and harness-doc checks passed for
the case-FPS time-scale fix. The full `tests/` collection was attempted but is
blocked by missing `trimesh` in the only local environment that currently has
`pytest`.

## Environment

- Date: 2026-06-24
- Machine: local WSL/Linux host
- Python: `Python 3.12.13`
- CUDA/GPU: two `NVIDIA GeForce RTX 4090`; GPU not required for focused tests
- Conda/env: `demo_2_max` for pytest and py_compile
- Commit: `2ba25d1` plus local task changes

## Commands Run

Red test before implementation:

```bash
conda run --no-capture-output -n demo_2_max python -m pytest tests/test_case_fps_time_scale.py -q
```

Result: failed as expected before implementation.

```text
7 failed
AttributeError: 'Config' object has no attribute 'apply_case_timing_from_metadata'
assert 0 == 1
```

Focused test after implementation:

```bash
conda run --no-capture-output -n demo_2_max python -m pytest tests/test_case_fps_time_scale.py -q
```

Result: passed.

```text
7 passed in 0.92s
```

Static compile:

```bash
conda run --no-capture-output -n demo_2_max python -m py_compile optimize_cma.py train_warp.py inference_warp.py qqtt/utils/config.py
```

Result: passed.

Harness docs:

```bash
python scripts/check_harness_docs.py
```

Result: passed.

```text
harness-docs: ok
```

Whitespace:

```bash
git diff --check
```

Result: passed.

Full local test attempt:

```bash
conda run --no-capture-output -n demo_2_max python -m pytest tests -q
```

Result: blocked by unrelated environment dependency in existing
`tests/test_data_process_sample_ground_policy.py`.

```text
7 passed, 2 failed
ModuleNotFoundError: No module named 'trimesh'
```

Environment dependency probes:

```bash
conda run --no-capture-output -n phystwin-max python - <<'PY'
for name in ['pytest', 'trimesh', 'open3d']:
    try:
        __import__(name)
        print(name, 'ok')
    except Exception as exc:
        print(name, 'missing', type(exc).__name__, exc)
PY
```

Result: `phystwin-max` has `trimesh` and `open3d` but not `pytest`.

```text
pytest missing ModuleNotFoundError No module named 'pytest'
trimesh ok
open3d ok
```

## Data Cases

- Case: synthetic metadata-only checks
- Input path: `tests/test_case_fps_time_scale.py`
- Output path: none

## Findings

- Severity: pending
- File: `optimize_cma.py`, `train_warp.py`, `inference_warp.py`
- Issue: real-case entrypoints read `metadata.json` but ignored `fps`, leaving
  `configs/real.yaml` timing active for 5 FPS Demo v4 cases.
- Evidence: focused red test failed before implementation; focused green test
  passed after adding shared config helper and entrypoint calls.
- Status: fixed for the scoped entrypoints.

## Acceptance Criteria Results

- Criterion: 5 FPS metadata yields 4000 substeps and 0.2 seconds per frame.
  Result: passed in `tests/test_case_fps_time_scale.py`.
- Criterion: missing FPS preserves configured fallback.
  Result: passed in `tests/test_case_fps_time_scale.py`.
- Criterion: invalid FPS fails early.
  Result: passed in `tests/test_case_fps_time_scale.py`.
- Criterion: three entrypoints call the shared helper.
  Result: passed in `tests/test_case_fps_time_scale.py`.

## Skipped Checks

- Check: full CMA/train/inference on GPU.
- Reason: focused request is the time-scale bug; full optimization is expensive
  and not needed to prove entrypoint config behavior.
- Next command to run: run case-specific optimization after user selects a real
  validation case.
- Check: full `pytest tests -q` in one environment.
- Reason: `demo_2_max` has `pytest` but lacks `trimesh`; `phystwin-max` and
  `demo_3_3_max` have `trimesh/open3d` but lack `pytest`.
- Next command to run: install `pytest` in `phystwin-max` or install `trimesh`
  in `demo_2_max`, then run `python -m pytest tests -q`.

## Residual Risk

- Risk: corrected 5 FPS cases run substantially slower because each observation
  transition uses 4000 substeps.
- Owner or follow-up: expected consequence of physical time correction.
