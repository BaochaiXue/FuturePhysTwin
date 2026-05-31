# Handoff

## Task ID

`2026-05-mvsam3d-shape-prior`

## Current State

2026-05-30 route split update: the production data-process route is now
single-view by default. Use `process_data_single_view.py
--single_view_backend sam3d` for the ordinary SAM3D path and
`--single_view_backend trellis` for the Trellis baseline. MV-SAM3D is an
explicit experimental route through `process_data_mvsam3d.py`. The old
`process_data_sam3d.py` and `script_process_data_sam3d.py` entrypoints are now
compatibility shims and default to the single-view SAM3D route.

Historical note: earlier work in this plan made MV-SAM3D the default backend.
That decision has been superseded because visual checks showed ordinary SAM3D
was usually cleaner for the tested cases.

The new adapter builds `shape/mvsam3d/input/images/<view>.png` and
`shape/mvsam3d/input/<mask_prompt>/<view>.png` RGBA masks from frame-0 case
data, resolving each camera's non-controller object through its own
`mask_info_<view>.json`. The wrapper prepares the scene, optionally runs DA3,
calls external `run_inference_weighted.py`, then normalizes canonical
`result.glb` back to `shape/object.glb`.

2026-05-29 audit update: wrapper paths are now resolved to absolute paths before
calling external MV-SAM3D, so direct commands using `--base_path ./data/...`
will not be interpreted relative to `MVSAM3D_ROOT`. `shape/object.glb` is also
validated as a non-empty mesh before the wrapper returns success. The wrapper
now refuses to substitute any non-`result.glb` artifact for `shape/object.glb`,
so debug/merged scene GLBs cannot silently change the downstream shape-prior
semantic contract.

2026-05-29 real-data update: the Hugging Face PhysTwin data zip was downloaded
to ignored `data/data.zip`, and `data/different_types/single_lift_sloth` was
extracted for a real check. MV-SAM3D generated a valid multi-view sloth prior
from views `0,1,2`. The full MV `result.glb` had 261,846 vertices / 523,672
faces; the downstream `shape/object.glb` is now simplified to 50,000 faces by
default while the full mesh remains under `shape/mvsam3d/outputs/`. This keeps
the downstream semantic as a single reconstructed object mesh and avoids making
`align.py` render a dense debug mesh.

`align.py` also needed to tolerate MV-SAM3D RGBA vertex colors when loading GLB
through PyTorch3D; `render_image()` now clips vertex features to RGB for
rendering. `align.py` and `data_process_sample.py` were switched from `avc1` to
`mp4v` MP4 output in this environment because H264 encoding was unavailable and
left stale videos.

2026-05-29 full MV align update: MV-SAM3D cases now have a dedicated
`data_process_sam3d/align_mvsam3d.py` path. `process_data_sam3d.py
--align_backend auto` sends `--shape_prior_backend mvsam3d` cases to this
multi-view aligner and keeps legacy/Trellis/SAM3D cases on `align.py`. The
aligner reads frame-0 manifest views, renders candidate views for each camera,
does SuperPoint/SuperGlue matching and PnP per view, chooses a robust shared
Sim(3) seed through multi-view reprojection/PCD/silhouette/depth scoring,
optimizes that Sim(3), then seeds ARAP deformation with keypoints and fused
object observations. Debug overlays and metrics are under
`shape/mvsam3d/align/`; compatible outputs remain
`shape/matching/final_mesh.glb` and `shape/matching/final_matching.mp4`.

`data_process_sam3d/data_process_sample.py` now also has an MV-SAM3D adaptive
sampling backend. In `auto` mode it selects the MV sampler when
`shape/mvsam3d/` exists; `process_data_sam3d.py` resolves this explicitly from
the shape-prior backend. On `single_lift_sloth`, this fixed the earlier
thin/empty MV final PCD by producing 700 surface and 1000 interior prior points.

2026-05-29 input-preprocess update: MV-SAM3D now defaults to the same
high-resolution pre-shape-prior preparation as the legacy single-view path, but
applied per selected view. `shape_prior_mvsam3d.py
--input_preprocess_backend legacy_upscale` runs `image_upscale.py` on each
view's masked object crop, then `segment_util_image.py` on the upscaled image,
and builds `shape/mvsam3d/input/` from those high-resolution RGB/RGBA files.
`raw` remains available as an explicit fallback. DA3 output is now namespaced by
preprocess backend, for example `shape/mvsam3d/da3/legacy_upscale/`, to avoid
reusing stale raw-input DA3 features.

2026-05-29 direct-runtime update: this machine uses `phystwin-max` as the
single direct runtime for PhysTwin and MV-SAM3D. The Python code no longer has
hardcoded named-conda-environment Python defaults. `process_data_sam3d.py`,
`script_process_data_sam3d.py`, and `shape_prior_mvsam3d.py` default
subprocesses to the current Python executable; the intended local invocation is
`/home/xinjie/miniforge3/envs/phystwin-max/bin/python ...`. Override flags
remain for non-standard setups only.

## Changed Files

- `data_process_sam3d/prepare_mvsam3d_scene.py`
- `data_process_sam3d/shape_prior_mvsam3d.py`
- `data_process_sam3d/align_mvsam3d.py`
- `process_data_routes.py`
- `process_data_single_view.py`
- `process_data_mvsam3d.py`
- `script_process_data_single_view.py`
- `script_process_data_mvsam3d.py`
- `process_data_sam3d.py`
- `script_process_data_sam3d.py`
- `data_process_sam3d/utils/align_util.py`
- `data_process_sam3d/align.py`
- `data_process_sam3d/data_process_sample.py`
- `README.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/task-brief.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/sprint-contract.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/qa-report.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/handoff.md`

## Verification Evidence

- Command:
  `python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py`
  Result: passed.
- Command:
  `python -m py_compile process_data_routes.py process_data_single_view.py process_data_mvsam3d.py script_process_data_single_view.py script_process_data_mvsam3d.py process_data_sam3d.py script_process_data_sam3d.py`
  Result: passed for the 2026-05-30 route split.
- Command: `python scripts/check_harness_docs.py`
  Result: passed, `harness-docs: ok`.
- Command: help checks for all changed CLIs.
  Result: passed.
- Command: synthetic two-view adapter smoke under `/tmp/fpt-mvsam3d-smoke`.
  Result: passed; RGB images, RGBA masks, manifest, and alpha `[0, 255]` were
  verified.
- Command: synthetic wrapper `--dry_run`.
  Result: passed; printed DA3 and MV-SAM3D inference commands.
- Command: relative `--base_path` dry-run from `/tmp/fpt-mvsam3d-relcheck`.
  Result: passed; external command paths were absolute and per-view object ids
  `1` and `2` were selected correctly.
- Command: dummy `normalize_outputs()` GLB validation under `phystwin-max`.
  Result: passed; copied `shape/object.glb` loaded as non-empty with 8 vertices
  and 12 faces.
- Command: dummy `normalize_outputs()` missing-`result.glb` negative check under
  `phystwin-max`.
  Result: passed; wrapper failed clearly before writing a substitute object GLB.
- Command: real adapter on `data/different_types/single_lift_sloth`.
  Result: passed; manifest selected `sloth` object index `0` in views `0,1,2`
  and wrote non-empty RGBA masks with alpha matching the source masks.
- Command: real MV-SAM3D wrapper on `single_lift_sloth` using
  `/home/xinjie/external/MV-SAM3D` and
  `/home/xinjie/miniforge3/envs/phystwin-max/bin/python`.
  Result: passed; wrote `shape/object.glb`, `shape/object.ply`, and debug
  outputs under `shape/mvsam3d/outputs/`.
- Command: `xvfb-run -a ... data_process_sam3d/align.py --force_rematch` on
  `single_lift_sloth`.
  Result: passed; 56 matches, reprojection error `8.973943710327148`, scale
  `0.35004712822820155`, and loadable `shape/matching/final_mesh.glb`.
- Command: `xvfb-run -a ... data_process_sam3d/data_process_sample.py
  --shape_prior` on `single_lift_sloth`.
  Result: passed; regenerated `final_data.pkl`, `final_pcd.mp4`, and
  `final_data.mp4`.
- Command: `xvfb-run -a ... data_process_sam3d/align_mvsam3d.py
  --force_rematch` on `single_lift_sloth`.
  Result: passed; `valid_views=2`, `total_inliers=41`,
  `best_initial_view=0`, distance metrics
  `obs_to_vertex median/p95=0.00204/0.00711 m` and
  `vertex_to_obs median/p95=0.00250/0.02992 m`; wrote
  `shape/matching/final_mesh.glb`, `shape/matching/final_matching.mp4`, and
  `shape/mvsam3d/align/metrics.json`.
- Command: `xvfb-run -a ... data_process_sam3d/data_process_sample.py
  --shape_prior --shape_prior_sampling_backend mvsam3d` on
  `single_lift_sloth`.
  Result: passed; regenerated `final_data.pkl`, `final_pcd.mp4`, and
  `final_data.mp4` with `surface_points (700, 3)` and
  `interior_points (1000, 3)`.
- Command: Trellis/original comparison rebuild from
  `shape/mvsam3d/original_shape_backup/matching/final_mesh.glb`.
  Result: passed; comparison artifacts were written to
  `/tmp/mvsam3d_vs_trellis_final_pcd.mp4` and
  `/tmp/mvsam3d_vs_trellis_final_pcd_contact.jpg`.
- Command: `/home/xinjie/miniforge3/envs/phystwin-max/bin/python
  data_process_sam3d/shape_prior_mvsam3d.py --input_preprocess_backend
  legacy_upscale --view_indices 0,1,2 --dry_run` on `single_lift_sloth`.
  Result: passed; planned six preprocessing commands, two per selected view,
  and DA3 output under `shape/mvsam3d/da3/legacy_upscale/da3_output.npz`.
- Command: direct-runtime dry-run under `phystwin-max`.
  Result: passed; the planned preprocess, DA3, and MV-SAM3D inference commands
  all used `/home/xinjie/miniforge3/envs/phystwin-max/bin/python`, with no
  hardcoded named-conda-environment Python command.

## Next Agent Should Read

1. `docs/plans/active/2026-05-mvsam3d-shape-prior/task-brief.md`
2. `docs/plans/active/2026-05-mvsam3d-shape-prior/sprint-contract.md`
3. `docs/plans/active/2026-05-mvsam3d-shape-prior/qa-report.md`
4. `data_process_sam3d/prepare_mvsam3d_scene.py`
5. `data_process_sam3d/shape_prior_mvsam3d.py`
6. `data_process_sam3d/align_mvsam3d.py`
7. `data_process_sam3d/data_process_sample.py`
8. `data_process_sam3d/utils/align_util.py`
9. `process_data_sam3d.py`
10. `script_process_data_sam3d.py`

## Open Follow-Ups

- Follow-up: create a dedicated MV-SAM3D conda env instead of continuing to add
  MV-SAM3D dependencies into the shared `phystwin-max` env.
- Follow-up: run full `process_data_sam3d.py` on `single_lift_sloth` only if
  segmentation/dense tracking/point-cloud stages need to be revalidated; the
  shape-prior, MV align, and final-data stages were already tested with existing
  case artifacts.
- Follow-up: tune `--mvsam3d_max_faces` per category if fine geometry is needed
  for matching; the verified default is 50,000 faces.
- Follow-up: run the MV align/sampling acceptance gates across more categories
  before calling MV-SAM3D the production default for every object class.

## Notes

- `shape/object.glb` intentionally comes from MV-SAM3D's canonical
  single-object `result.glb`; merged DA3 scene outputs are kept as debug
  artifacts only.
- `align_mvsam3d.py` requires CUDA/PyTorch3D and fails clearly when CUDA is not
  available. In the verified sloth run it had two valid views; only view `0`
  was used as a strong ARAP keypoint source after robust scoring because view
  `2` was geometrically weaker, while all manifest views still contributed to
  loading, scoring, diagnostics, and fused observation constraints.
- Full high-resolution MV-SAM3D inference was rerun on `single_lift_sloth`
  with views `0,1,2` and `legacy_upscale` inputs. It regenerated
  `shape/object.glb`, `shape/object.ply`, `shape/matching/final_mesh.glb`,
  `final_data.pkl`, `final_pcd.mp4`, and `final_data.mp4`.
- The rerun passed the final-data point-count gate with `surface_points (700, 3)`
  and `interior_points (1000, 3)`, but align metrics still show residual
  overgrowth risk: `vertex_to_obs p95=0.05025 m`, above the 35 mm target.
- The DA3 wrapper now supports `--da3_model_path`, with
  `--mvsam3d_da3_model_path` pass-through, for Hugging Face snapshot cache
  layouts.
- A later MV align-quality fix changed `align_mvsam3d.py` to select a
  gate-aware `sim3_only` result when ARAP/ray-ARAP would overgrow the mesh. On
  `single_lift_sloth`, the forced rematch run now passes with
  `vertex_to_obs p95=0.03042 m` and `obs_to_vertex p95=0.01394 m`.
- Corrected clean rerun evidence lives under
  `/tmp/fpt-single_lift_sloth-rerun-20260530-000420`. This run used fresh
  Trellis and MV-SAM3D case copies from the same observations, not
  `original_shape_backup` or any ambiguous backup path. MV align acceptance was
  observation-only: `valid_views=2`, `total_inliers=38`,
  `best_initial_view=2`, `selected_scale_multiplier=0.95`,
  `vertex_to_obs p95=0.03171 m`, `obs_to_vertex p95=0.01450 m`, and
  `quality_gates.passed=true`.
- The corrected scoreboard is
  `/tmp/fpt-single_lift_sloth-rerun-20260530-000420/scoreboard.json`.
  Presentation videos are
  `/tmp/fpt-single_lift_sloth-rerun-20260530-000420/artifacts/LEFT_TRELLIS_RIGHT_MVSAM3D_final_pcd.mp4`
  and
  `/tmp/fpt-single_lift_sloth-rerun-20260530-000420/artifacts/LEFT_TRELLIS_RIGHT_MVSAM3D_raw_mesh.mp4`.
  The scoreboard reports `mv_beats_trellis.passed=true`, but this is only the
  final end-to-end baseline comparison; it is not used as MV align evidence.
- MV-SAM3D final sampling now caps its effective shape-prior distance filter at
  `0.035 m`; the verified rerun kept `surface_points (700, 3)` and
  `interior_points (1000, 3)`.
- A second isolated clean rerun after the artifact-mix concern lives under
  `/tmp/fpt-single_lift_sloth-clean-rerun-20260530-011819`. It has separate
  backend case copies at `trellis_case/single_lift_sloth` and
  `mvsam3d_case/single_lift_sloth`, plus copied labeled artifacts under
  `artifacts/`. The key presentation videos are
  `/tmp/fpt-single_lift_sloth-clean-rerun-20260530-011819/artifacts/LEFT_TRELLIS_RIGHT_MVSAM3D_final_pcd.mp4`
  and
  `/tmp/fpt-single_lift_sloth-clean-rerun-20260530-011819/artifacts/LEFT_TRELLIS_RIGHT_MVSAM3D_raw_mesh.mp4`.
  MV align remained observation-only and passed with `valid_views=2`,
  `total_inliers=43`, `selected_stage=sim3_only`, `vertex_to_obs p95=0.03060 m`,
  and `obs_to_vertex p95=0.01202 m`. The scoreboard reports MV final
  pcd-to-observation p95 `0.00325 m` versus Trellis `0.01158 m`.
- Ordinary single-view SAM3D was also rerun in isolation under
  `/tmp/fpt-single_lift_sloth-sam3d-rerun-20260530-021056`. Artifacts are under
  `artifacts/` with `SAM3D_` prefixes. The most useful viewing files are
  `SAM3D_final_pcd_labeled.mp4`, `SAM3D_raw_mesh_labeled.mp4`,
  `LEFT_SAM3D_RIGHT_MVSAM3D_final_pcd.mp4`, and
  `LEFT_SAM3D_RIGHT_MVSAM3D_raw_mesh.mp4`. The SAM3D baseline produced
  `surface_points (570, 3)`, `interior_points (990, 3)`,
  final-to-observation p95 `0.02137 m`, observation-to-final p95 `0.00415 m`,
  and prior far fraction `0.05321` beyond `35 mm`.
- The batch wrapper now follows the safer `script_process_data.py` style:
  `case_filter`, checked subprocess calls, and list-based commands. This also
  fixes unquoted categories such as `stuffed animal`.
- Existing unrelated local changes were left untouched:
  `AGENTS.md`, `gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`,
  and `docs/plans/active/2026-05-24-phystwin-max-env/`.
- Ignored local runtime assets now include `data/data.zip`,
  `data/different_types/single_lift_sloth/`,
  `data_process_sam3d/models/weights/`, and external checkouts under
  `/home/xinjie/external/`.
- Route split implementation added `process_data_single_view.py` and
  `process_data_mvsam3d.py`, plus matching batch scripts. Ordinary SAM3D and
  Trellis should be compared through the single-view route because it follows
  `data_process` preprocessing, legacy align, and legacy final sampling.
  MV-SAM3D should be compared through the explicit MV route only.
- Compatibility shims still accept old `process_data_sam3d.py` commands, but
  they reject incompatible requests such as single-view SAM3D with
  `--align_backend mvsam3d`.
- A fresh functional check of the single-view SAM3D route passed under
  `/tmp/fpt-single-view-sam3d-route-check-20260530-172406`. Its route manifest
  records `route=single_view`, `backend=sam3d`, and the expected stage order:
  `data_process` video segmentation/upscale/image segmentation, SAM3D shape
  prior, `data_process` dense tracking/pcd/mask/track, legacy align, and legacy
  final sampling. Outputs include loadable `shape/object.glb`, loadable
  `shape/matching/final_mesh.glb`, `final_data.pkl`, `final_pcd.mp4`, and
  `final_data.mp4`.
- The rerun required compatibility fixes now present in the tree: segmentation
  subprocesses use `sys.executable` and checked subprocess calls, video
  segmentation has the BERT `get_head_mask` compatibility shim, single-object
  cases tolerate one controller detection, SAM3D root/config resolution can use
  `/home/xinjie/external/MV-SAM3D`, and legacy align warns/falls back when
  Open3D ARAP factorization fails instead of aborting the pipeline.
