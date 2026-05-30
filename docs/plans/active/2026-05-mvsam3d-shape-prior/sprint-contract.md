# Sprint Contract

## Contract ID

`mvsam3d-shape-prior-backend`

## Goal

Replace the SAM3D single-view shape-prior stage with a default MV-SAM3D
multi-view backend and route MV-SAM3D cases through an MV-specific multi-view
align and sampling path, while keeping the old SAM3D/Trellis path as an
explicit fallback.

## Agreed Scope

Files expected to change:

- `data_process_sam3d/prepare_mvsam3d_scene.py`
- `data_process_sam3d/shape_prior_mvsam3d.py`
- `data_process_sam3d/align_mvsam3d.py`
- `data_process_sam3d/utils/align_util.py`
- `data_process_sam3d/align.py`
- `data_process_sam3d/data_process_sample.py`
- `process_data_sam3d.py`
- `script_process_data_sam3d.py`
- `README.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/qa-report.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/handoff.md`

Files expected to read:

- `AGENTS.md`
- `HARNESS.md`
- `docs/harness/README.md`
- `docs/harness/operating-model.md`
- `docs/harness/verification.md`
- `data_process_sam3d/shape_prior.py`
- `data_process_sam3d/align.py`
- `data_process_sam3d/align_mvsam3d.py`
- `data_process_sam3d/data_process_sample.py`
- `process_data.py`
- `script_process_data.py`

Out of scope:

- Rewriting segmentation, tracking, point-cloud lifting, mask post-processing,
  controller logic, or upstream case generation.
- Adding multi-object downstream support.
- Vendoring MV-SAM3D, DA3, checkpoints, tokens, caches, or generated assets.

## Behavior To Deliver

- Build an MV-SAM3D scene from all available frame-0 camera views, using each
  camera's own `mask_info_<cam_idx>.json` to select exactly one non-controller
  object.
- By default, mirror the legacy SAM3D pre-shape-prior preparation for each
  selected MV-SAM3D view: masked object crop upscale through `image_upscale.py`,
  followed by `segment_util_image.py` to produce a high-resolution RGBA mask.
  Keep `raw` as an explicit fallback input mode.
- Write RGBA object masks whose alpha channel is foreground and record source
  and generated paths in `shape/mvsam3d/manifest.json`.
- Run external MV-SAM3D through `--mvsam3d_root`/`MVSAM3D_ROOT` and
  `--mvsam3d_python`/`MVSAM3D_PYTHON`, with optional DA3 generation.
- Normalize canonical MV-SAM3D object output to `shape/object.glb` and
  optional `shape/object.ply`/`shape/visualization.mp4`.
- Keep the full MV-SAM3D object mesh as a debug artifact and simplify the
  downstream `shape/object.glb` to `--mvsam3d_max_faces` by default so
  `align.py` receives the same single-object mesh semantic without excessive
  render cost.
- Preserve existing segmentation, tracking, and 3D processing behavior after
  the shape-prior stage.
- Add `data_process_sam3d/align_mvsam3d.py` for MV-SAM3D priors. It uses
  frame-0 manifest views for multi-view matching, PnP candidate scoring,
  shared Sim(3) optimization with reprojection/PCD/silhouette/depth terms, and
  ARAP deformation against fused observations.
- Keep `align.py` intact and available through `--align_backend legacy`.
- Add MV-SAM3D adaptive sampling in `data_process_sample.py` so final data has
  enough accepted surface/interior prior points after grid filtering.

## Acceptance Criteria

- Must: changed Python CLIs compile and expose expected help flags.
- Must: dry-run mode reports the converter and external commands without
  requiring MV-SAM3D imports.
- Must: default `--shape_prior_backend` is `mvsam3d`; `sam3d` retains the old
  image-upscale/masked-image/`shape_prior.py` path.
- Must not: modify `data_config.csv` format or commit generated shape assets.
- Must not: silently use merged DA3 scene GLBs as `shape/object.glb`.
- Must: real MV-SAM3D GLBs with RGBA vertex colors remain renderable by
  `align.py`.
- Must: MV-SAM3D DA3 cache is separated by input preprocess backend so raw and
  high-resolution scene inputs cannot silently share `da3_output.npz`.
- Must: `--align_backend auto` routes MV-SAM3D priors to `align_mvsam3d.py` and
  legacy priors to `align.py`.
- Must: on `single_lift_sloth`, MV-SAM3D final data contains at least 700
  surface points and 1000 interior points after MV sampling.

## Evaluator Probes

The evaluator should explicitly check:

- Adapter handles per-view object-id resolution and rejects missing or empty
  masks with actionable errors.
- Process pipeline skips the single-view image-upscale/segment-util steps for
  the MV-SAM3D backend.
- Batch script passes MV-SAM3D flags only when relevant and keeps case filtering
  compatible with existing `data_config.csv`.
- QA report records real GPU/MV-SAM3D evidence when the runtime exists, or
  skipped checks plus exact follow-up commands when it does not.
- MV align metrics include valid views, inlier counts, distance metrics, output
  paths, and any view-gating decision used for robust keypoint constraints.

## Verification Commands

```bash
python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py data_process_sam3d/align_mvsam3d.py data_process_sam3d/data_process_sample.py data_process_sam3d/utils/align_util.py
python scripts/check_harness_docs.py
python data_process_sam3d/prepare_mvsam3d_scene.py --help
python data_process_sam3d/shape_prior_mvsam3d.py --help
python data_process_sam3d/align_mvsam3d.py --help
python data_process_sam3d/data_process_sample.py --help
python process_data_sam3d.py --help
python script_process_data_sam3d.py --help
```

Add a local-case dry-run when case data is available:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name <case> \
  --category <category> \
  --controller_name hand \
  --dry_run
```

Functional GPU follow-up:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python process_data_sam3d.py \
  --base_path ./data/different_types \
  --case_name <case> \
  --category <category> \
  --shape_prior \
  --shape_prior_backend mvsam3d \
  --mvsam3d_run_da3 \
  --mvsam3d_da3_model_path <optional-da3-snapshot-or-checkout>
```

MV align/sampling functional check:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/align_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --controller_name hand \
  --force_rematch
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/data_process_sample.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --shape_prior \
  --shape_prior_sampling_backend mvsam3d
```

## Contract Changes

- 2026-05-24: Initial contract from the user request. The batch script is also
  allowed to adopt the safer `script_process_data.py` subprocess/case-filter
  style because the user asked to fix inconsistencies with the original data
  processing part.
- 2026-05-29: Real-data verification exposed two downstream compatibility
  issues: dense MV meshes were too expensive for `align.py`, and PyTorch3D saw
  RGBA vertex colors. The contract now includes default mesh simplification and
  an align renderer guard while still preserving the single-object shape-prior
  semantic.
- 2026-05-29: User requested a full MV-SAM3D align rewrite after the first
  `single_lift_sloth` comparison looked thinner than Trellis. Scope expanded to
  include `align_mvsam3d.py`, `--align_backend auto`, MV adaptive final-data
  sampling, and Trellis-vs-MV comparison artifacts.
- 2026-05-29: User requested that MV-SAM3D also use the legacy high-resolution
  pre-shape-prior step. Scope expanded to include per-view
  `legacy_upscale` input preprocessing before MV-SAM3D inference.
