# Task Brief

## Task ID

`2026-05-mvsam3d-shape-prior`

## Request

Replace the current single-view SAM3D shape-prior stage in the SAM3D data
processing path with an MV-SAM3D multi-view backend, then give that backend its
own multi-view alignment and sampling path while preserving the downstream
PhysTwin data-processing contract.

## Outcome

When `--shape_prior` is requested, `process_data_sam3d.py` should default to
building an MV-SAM3D scene from all available calibrated frame-0 camera views,
run the external MV-SAM3D backend when configured, align the generated object
with the MV-specific aligner, and normalize the final outputs back to the
existing shape/data paths used by downstream PhysTwin code.

## Scope

- In scope:
  - A lightweight FuturePhysTwin-to-MV-SAM3D input adapter.
  - A runtime wrapper around an external MV-SAM3D checkout.
  - An MV-SAM3D-specific multi-view alignment backend.
  - Adaptive mesh-guided final-data sampling for MV-SAM3D shape priors.
  - CLI flags in `process_data_sam3d.py` and `script_process_data_sam3d.py`.
  - Documentation, static checks, help checks, and dry-run evidence.
- Out of scope:
  - Segmentation model changes.
  - Dense tracking, point-cloud lifting, and mask post-processing.
  - Controller logic changes beyond exposing the current `hand` default.
  - Multi-object downstream support.
  - Gaussian training, simulator changes, and pose/physics optimization.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/README.md`
- `docs/harness/operating-model.md`
- `docs/harness/verification.md`
- `process_data_sam3d.py`
- `script_process_data_sam3d.py`
- `data_process_sam3d/shape_prior.py`
- `data_process_sam3d/align.py`
- `data_process_sam3d/align_mvsam3d.py`
- `data_process_sam3d/data_process_sample.py`
- `docs/plans/completed/2026-05-10-sam3d-runtime-test/qa-report.md`
- `docs/plans/completed/2026-05-10-sam3d-runtime-test/handoff.md`

## Data And Hardware Assumptions

- Required data: a case under `<base_path>/<case_name>/` with
  `color/<cam_idx>/0.png`, `mask/mask_info_<cam_idx>.json`, and
  `mask/<cam_idx>/<obj_idx>/0.png`.
- Required device: CPU is sufficient for adapter dry-runs; GPU is required for
  real MV-SAM3D and DA3 inference.
- Expected runtime: adapter dry-runs should finish quickly; functional MV-SAM3D
  runtime depends on checkpoint and GPU state.
- External services: MV-SAM3D and any checkpoints remain outside this repo and
  are located by `MVSAM3D_ROOT`/`MVSAM3D_PYTHON` or equivalent CLI flags.

## Acceptance Criteria

- `data_process_sam3d/prepare_mvsam3d_scene.py` creates or dry-runs a scene
  with `images/<view>.png`, `<object_name>/<view>.png` RGBA masks, and
  `manifest.json`.
- `data_process_sam3d/shape_prior_mvsam3d.py` can dry-run without importing
  MV-SAM3D and can call external DA3/MV-SAM3D when configured.
- `process_data_sam3d.py` keeps the legacy SAM3D backend available and defaults
  the requested shape-prior backend to `mvsam3d`.
- `process_data_sam3d.py --align_backend auto` sends MV-SAM3D priors to
  `align_mvsam3d.py` and leaves legacy/Trellis/SAM3D priors on `align.py`.
- `data_process_sam3d/data_process_sample.py` provides an MV-SAM3D sampler that
  avoids thin or under-filled `final_pcd` output on `single_lift_sloth`.
- `script_process_data_sam3d.py` passes backend and MV-SAM3D flags without
  requiring a `data_config.csv` format change.
- Downstream output paths remain `shape/object.glb`, optional
  `shape/object.ply`, optional `shape/visualization.mp4`,
  `shape/matching/final_mesh.glb`, `final_data.pkl`, `final_pcd.mp4`, and
  `final_data.mp4`.

## Verification Plan

- Static checks:
  - `python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py data_process_sam3d/align_mvsam3d.py data_process_sam3d/data_process_sample.py data_process_sam3d/utils/align_util.py`
  - `python scripts/check_harness_docs.py`
- Functional checks:
  - help output for changed CLIs.
  - adapter/wrapper dry-run on a local case when one exists.
- Full checks:
  - run one MV-SAM3D-backed shape-prior case when external MV-SAM3D, DA3,
    checkpoints, and GPU are available.
  - run `align_mvsam3d.py` and MV sampling on `single_lift_sloth`, then compare
    MV-SAM3D and Trellis/original final PCD density.

## Risks

- Risk: external MV-SAM3D/DA3 runtime may not be installed locally.
  Mitigation: keep dry-run and static checks meaningful, and record the exact
  functional command as a follow-up if skipped.
- Risk: MV-SAM3D merged-scene outputs can mix coordinate systems with the
  object mesh expected by `align.py`.
  Mitigation: normalize `shape/object.glb` from the canonical object result and
  keep merged/optimized scene files only as debug artifacts.
- Risk: object ids can vary by camera.
  Mitigation: resolve each camera through its own `mask_info_<cam_idx>.json`
  and fail clearly if the v1 single-object contract is violated.
