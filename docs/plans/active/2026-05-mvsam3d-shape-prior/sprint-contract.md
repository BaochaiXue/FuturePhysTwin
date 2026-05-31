# Sprint Contract

## Contract ID

`shape-prior-route-split`

## Goal

Split the current mixed SAM3D/MV-SAM3D data-process entrypoint into a
production single-view route and an explicit experimental MV-SAM3D route.
Ordinary SAM3D and Trellis should mainly follow the original `data_process`
semantics; MV-SAM3D should no longer be the default or share ambiguous route
flags with single-view backends.

## Files Expected To Change

- `process_data_routes.py`
- `process_data_single_view.py`
- `process_data_mvsam3d.py`
- `script_process_data_single_view.py`
- `script_process_data_mvsam3d.py`
- `process_data_sam3d.py`
- `script_process_data_sam3d.py`
- `README.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/task-brief.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/sprint-contract.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/qa-report.md`
- `docs/plans/active/2026-05-mvsam3d-shape-prior/handoff.md`

Existing MV implementation files may remain dirty from prior align/sampling QA,
but this route split should not rewrite them except for verification-driven
fixes.

## Behavior To Deliver

- Add `process_data_single_view.py`:
  - `--single_view_backend {sam3d,trellis}`, default `sam3d`.
  - Uses `data_process/segment.py`, `image_upscale.py`,
    `segment_util_image.py`, `dense_track.py`, `data_process_pcd.py`,
    `data_process_mask.py`, `data_process_track.py`, `align.py`, and
    `data_process_sample.py`.
  - Uses `data_process/shape_prior.py` for Trellis and
    `data_process_sam3d/shape_prior.py` for SAM3D.
  - Has no MV-SAM3D CLI flags.
- Add `process_data_mvsam3d.py`:
  - Uses `data_process_sam3d/shape_prior_mvsam3d.py`,
    `data_process_sam3d/align_mvsam3d.py`, and
    `data_process_sam3d/data_process_sample.py --shape_prior_sampling_backend
    mvsam3d`.
  - Defaults to frame-0 views `0,1,2` and `legacy_upscale`.
  - Has no single-view backend switch.
- Add matching batch scripts for each route.
- Convert `process_data_sam3d.py` and `script_process_data_sam3d.py` to
  compatibility shims:
  - default `--shape_prior_backend sam3d`;
  - `sam3d` forwards to the single-view route;
  - `mvsam3d` forwards to the MV route;
  - incompatible legacy flags fail clearly.
- Write `shape/route_manifest.json` for every run with route, backend, command
  records, output paths, and split metadata.

## Acceptance Criteria

- Single-view SAM3D/Trellis output semantics match the original `data_process`
  route except for the selected shape-prior generator.
- MV-SAM3D remains isolated under its own entrypoint and debug directory.
- No code path makes MV-SAM3D the default backend.
- `data_config.csv` remains unchanged.
- Public downstream outputs remain compatible with existing consumers:
  `shape/object.glb`, `shape/matching/final_mesh.glb`, `final_data.pkl`,
  `final_pcd.mp4`, `final_data.mp4`, and `split.json`.

## Verification Commands

```bash
python -m py_compile process_data_routes.py process_data_single_view.py process_data_mvsam3d.py script_process_data_single_view.py script_process_data_mvsam3d.py process_data_sam3d.py script_process_data_sam3d.py
python process_data_single_view.py --help
python process_data_mvsam3d.py --help
python script_process_data_single_view.py --help
python script_process_data_mvsam3d.py --help
python process_data_sam3d.py --help
python script_process_data_sam3d.py --help
python scripts/check_harness_docs.py
git diff --check
```

Functional follow-up, when runtime and case data are available:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python process_data_single_view.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --shape_prior \
  --single_view_backend sam3d

/home/xinjie/miniforge3/envs/phystwin-max/bin/python process_data_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --shape_prior \
  --mvsam3d_view_indices 0,1,2 \
  --mvsam3d_input_preprocess_backend legacy_upscale
```
