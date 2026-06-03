# Sprint Contract

## Contract ID

`demo33-shape-prior-warmup-speed`

## Goal

Reduce Demo 3.3 shape-prior completion wall time without reducing quality or
skipping required shape-prior stages.

## Agreed Scope

Files expected to change:

- `data_process/align.py`
- `data_process_sam3d/shape_prior.py`
- `data_process_sam3d/data_process_sample.py`
- `/home/xinjie/proj-QQTT-v2/qqtt/demo/demo33_shape_prior_warmup.py`
- `/home/xinjie/proj-QQTT-v2/tests/test_demo33_shape_prior_warmup.py`

Files expected to read:

- `data_process/utils/align_util.py`
- `data_process/match_pairs.py`
- `/home/xinjie/proj-QQTT-v2/qqtt/demo/demo33_shape_prior_completion.py`

Out of scope:

- Lowering SAM3D inference settings or alignment pose/matching hyperparameters.
- Removing shape-prior generation, alignment, sampling, final-data validation,
  or render-layer attach.

## Behavior To Deliver

- Batch align ray visibility tests instead of per-vertex ray calls where
  possible.
- Keep SAM3D mesh export, data sampling, and `final_data.pkl` generation, but
  let Demo 3.3 avoid optional turntable visualization videos.
- Record verification evidence.

## Acceptance Criteria

- Must: preserve route command order and required stages.
- Must: preserve final-data fields and shape-prior coordinate policy.
- Must not: reduce point counts, image size, SAM3D settings, matching thresholds,
  or alignment pose counts.

## Evaluator Probes

The evaluator should explicitly check:

- Probe: Demo 3.3 route still invokes `data_process_sample.py --shape_prior`.
- Probe: `--skip_visualization` only skips videos, not `final_data.pkl`.
- Probe: align batching uses the same radius/visibility thresholds.

## Verification Commands

```bash
python scripts/check_harness_docs.py
python -m py_compile data_process/align.py data_process_sam3d/shape_prior.py data_process_sam3d/data_process_sample.py
```

QQTT-side:

```bash
conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo33_shape_prior_warmup.py -q
```

## Contract Changes

- 2026-06-03: Initial contract from user request to improve warmup speed without
  quality loss.
