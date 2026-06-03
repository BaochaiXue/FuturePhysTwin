# QA Report

## Task

`2026-06-03-demo33-shape-prior-warmup-speed`

## Summary

Demo 3.3 shape-prior completion now preserves the full required route while
reducing route wall time. The main speedups are:

- `data_process/align.py`: batched mesh ray visibility checks and KD-tree
  observation nearest-neighbor lookup replace per-vertex Python loops.
- `data_process_sam3d/data_process_sample.py`: `--skip_visualization` skips
  optional `final_pcd.mp4` / `final_data.mp4` generation only.
- `data_process_sam3d/shape_prior.py`: `--skip_visualization` skips optional
  `shape/visualization.mp4` only; `object.glb` and `object.ply` are still
  exported.
- QQTT Demo 3.3 passes the visualization-skip flag by default through
  `--shape-prior-skip-route-visualizations`.

No SAM3D inference settings, alignment pose counts, matching thresholds, image
size, point targets, route stages, coordinate policy, or render-layer attach
were removed.

## Verification Commands

```bash
python -m py_compile data_process_sam3d/shape_prior.py data_process/align.py data_process_sam3d/data_process_sample.py
python scripts/check_harness_docs.py
```

Result:

```text
harness-docs: ok
```

QQTT-side static and focused checks:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile qqtt/demo/demo33_shape_prior_warmup.py tests/test_demo33_shape_prior_warmup.py
conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo33_shape_prior_warmup.py tests/test_demo31_dual_gpu_contract.py::Demo31DualGpuContractTest::test_demo33_dry_run_contract_enables_shape_prior_warmup tests/test_demo31_dual_gpu_contract.py::Demo31DualGpuContractTest::test_demo33_shape_prior_warmup_starts_async_without_blocking_tracker_input -q
```

Result:

```text
9 passed
```

QQTT full quick harness:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
```

Result:

```text
Ran 389 tests
OK
quick deterministic checks passed
```

## Runtime Evidence

Baseline profile before this optimization:

```text
docs/generated/demo33_after_teardown_detached_shape_prior_20260603_60s_profile_shape_prior_completion.json
shape_prior_warmup.shape_prior_async_elapsed_ms = 193752
image_upscale = 18745 ms
segment_util_image = 8902 ms
shape_prior_sam3d = 39244 ms
align = 81402 ms
data_process_sample = 45445 ms
```

Optimized detached completion smoke, downstream visualization skip:

```text
docs/generated/demo33_shape_prior_speed_smoke_20260603_completion.json
shape_prior_warmup.shape_prior_async_elapsed_ms = 108273
image_upscale = 18785 ms
segment_util_image = 9038 ms
shape_prior_sam3d = 39309 ms
align = 31415 ms
data_process_sample = 9714 ms
shape_prior_status = ready
shape_prior_coordinate_validation_status = valid
shape_prior_render_layer_enabled = true
shape_prior_structure_points = 9735
```

Optimized detached completion smoke, including SAM3D visualization skip:

```text
docs/generated/demo33_shape_prior_speed_smoke_skip_sam3d_vis_20260603_completion.json
shape_prior_warmup.shape_prior_async_elapsed_ms = 118988
image_upscale = 18773 ms
segment_util_image = 9510 ms
shape_prior_sam3d = 43560 ms
align = 36503 ms
data_process_sample = 10631 ms
shape_prior_status = ready
shape_prior_coordinate_validation_status = valid
shape_prior_render_layer_enabled = true
shape_prior_structure_points = 9727
```

Stage-level direct smoke on the same case:

```text
align --skip_visualization: 15.86 s
data_process_sample --shape_prior --ground-policy preserve --skip_visualization: 10.29 s
final_data load status: ready
coordinate_validation_status: valid
render_layer_enabled: true
```

## Interpretation

The route is no longer 3-minute-plus on the validated case. The representative
optimized completion range is roughly 108-119 seconds, with stochastic SAM3D,
segmentation, and SuperGlue/align variation between runs.

Skipping SAM3D `visualization.mp4` is a correct flow cleanup but not the primary
speed source in the measured run. The main wins are the align batching/KD-tree
change and skipping downstream diagnostic videos that do not feed
`final_data.pkl` or the Demo 3.3 render layer.

## Residual Risks

- SAM3D still attempts gsplat CUDA JIT layout post optimization and falls back
  because `phystwin-max` has no `nvcc`; this is caught by SAM3D and does not
  block readiness, but it is still environment noise.
- A fresh case may differ from the reused validation case because SuperGlue
  match count and SAM3D mesh topology vary.
- Further speedups without quality loss likely require process-level model
  reuse or upstream SAM3D route consolidation, not reduced hyperparameters.
