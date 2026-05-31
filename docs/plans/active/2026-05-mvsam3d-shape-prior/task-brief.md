# Task Brief

## Task ID

`2026-05-mvsam3d-shape-prior`

## Request

Reorganize the data-process shape-prior flow into two explicit routes instead
of mixing ordinary SAM3D and MV-SAM3D behind the same `process_data_sam3d.py`
flags.

## Outcome

The production route is now single-view and `data_process`-compatible:
Trellis and SAM3D share the original preprocessing, alignment, and final-data
sampling semantics, and only differ in the shape-prior generator. MV-SAM3D is an
explicit experimental route with its own multi-view adapter, aligner, sampler,
debug directory, and manifest lineage.

## Scope

- In scope:
  - New single-view entrypoints for `sam3d` and `trellis`.
  - New MV-SAM3D entrypoints for the experimental multi-view route.
  - Compatibility shims for the old SAM3D entrypoints.
  - Route manifests that identify `single_view/sam3d`,
    `single_view/trellis`, or `mvsam3d/mvsam3d`.
  - README, sprint contract, QA report, and handoff updates.
- Out of scope:
  - Rewriting segmentation, tracking, point-cloud lifting, or mask
    post-processing.
  - Changing the `data_config.csv` schema.
  - Vendoring external MV-SAM3D, DA3, SAM3D, Trellis, checkpoints, caches, or
    generated assets.

## Route Contract

- `process_data_single_view.py --single_view_backend sam3d` is the production
  default.
- `process_data_single_view.py --single_view_backend trellis` is the original
  Trellis baseline route.
- Single-view routes use `data_process/*` for preprocessing, segmentation,
  dense tracking, lifting, mask post-processing, legacy `align.py`, and legacy
  `data_process_sample.py`.
- SAM3D single-view only swaps the shape-prior generator to
  `data_process_sam3d/shape_prior.py`.
- `process_data_mvsam3d.py` is explicit and experimental. It uses
  `shape_prior_mvsam3d.py`, `align_mvsam3d.py`, and MV sampling.
- `process_data_sam3d.py` and `script_process_data_sam3d.py` remain
  compatibility shims and default to the single-view SAM3D route.

## Acceptance Criteria

- The new CLIs compile and expose help output.
- The single-view route accepts no `mvsam3d_*` flags and never invokes
  `align_mvsam3d.py`.
- The MV route has no `--shape_prior_backend`, `--align_backend`, or
  `--shape_prior_sampling_backend` public switch; it uses the MV backends
  internally.
- Each route writes `shape/route_manifest.json` with route/backend labels and
  exact commands.
- Existing downstream public outputs remain:
  `shape/object.glb`, `shape/matching/final_mesh.glb`, `final_data.pkl`,
  `final_pcd.mp4`, `final_data.mp4`, and `split.json`.

## Verification Plan

- Static:
  - `python -m py_compile process_data_routes.py process_data_single_view.py process_data_mvsam3d.py script_process_data_single_view.py script_process_data_mvsam3d.py process_data_sam3d.py script_process_data_sam3d.py`
  - `python scripts/check_harness_docs.py`
  - `git diff --check`
- CLI:
  - help output for both new route scripts, both batch scripts, and both
    compatibility shims.
- Functional when runtime/data are available:
  - run one `single_view/sam3d` case.
  - run one `single_view/trellis` case if Trellis weights are available.
  - run one explicit `mvsam3d` case and verify the manifest and MV metrics.
