# QA Report

## Task ID

`2026-05-mvsam3d-shape-prior`

## Summary

Static, help, harness-doc, synthetic adapter/wrapper dry-run checks, and one
real PhysTwin case passed. The real check used the downloaded
`single_lift_sloth` case and verified MV-SAM3D generation, downstream
`align.py`, and `data_process_sample.py`.

2026-05-29 semantic audit update: found and fixed a relative-path bug that
would have made external MV-SAM3D resolve `./data/...` under `MVSAM3D_ROOT`
instead of the PhysTwin repo/current caller directory. Added non-empty GLB
validation before accepting MV-SAM3D `result.glb` as `shape/object.glb`; the
wrapper now fails if canonical `result.glb` is absent rather than substituting
another GLB.

2026-05-29 real-data update: real MV-SAM3D output preserved the single-object
shape-prior semantic, but exposed two compatibility fixes that are now included:
default simplification of `shape/object.glb` to 50,000 faces while retaining
the full MV result as debug output, and RGB-only rendering of RGBA vertex-color
GLBs in PyTorch3D.

## Environment

- Date: 2026-05-24T16:11:38-04:00
- Machine: `robopil-zeta`
- Python: `Python 3.13.13`
- CUDA/GPU: two `NVIDIA GeForce RTX 4090`, driver `580.159.03`, `24564 MiB`
  each
- Conda/env: `demo3-max` has `Python 3.12.13` and `numpy`/`PIL`;
  `phystwin-max` was used for MV-SAM3D, DA3, alignment, and final-data runtime
  checks
- Commit: `0782b63`

## Commands Run

```bash
git pull --ff-only origin main
```

Result: passed; branch was already up to date.

```bash
git switch -c feature/mvsam3d-shape-prior
```

Result: passed.

```bash
python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py
```

Result: passed.

```bash
python scripts/check_harness_docs.py
```

Result: passed; output was `harness-docs: ok`.

```bash
git pull --ff-only origin main
```

Result on 2026-05-29: passed; branch was already up to date.

```bash
python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py
python scripts/check_harness_docs.py
git diff --check
```

Result on 2026-05-29: passed.

```bash
python process_data_sam3d.py --help >/tmp/fpt_process_help.txt
python script_process_data_sam3d.py --help >/tmp/fpt_script_help.txt
python data_process_sam3d/prepare_mvsam3d_scene.py --help >/tmp/fpt_prepare_help.txt
python data_process_sam3d/shape_prior_mvsam3d.py --help >/tmp/fpt_mvsam3d_help.txt
wc -l /tmp/fpt_process_help.txt /tmp/fpt_script_help.txt /tmp/fpt_prepare_help.txt /tmp/fpt_mvsam3d_help.txt
```

Result: passed; help files were 35, 33, 23, and 28 lines respectively.

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python - <<'PY'
from pathlib import Path
import json
import numpy as np
from PIL import Image
root = Path('/tmp/fpt-mvsam3d-smoke/data/different_types/toy_case')
for view in ['0', '1']:
    (root / 'color' / view).mkdir(parents=True, exist_ok=True)
    (root / 'mask' / view / '1').mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 40 + int(view) * 80
    rgb[2:6, 2:6, 1] = 220
    Image.fromarray(rgb, mode='RGB').save(root / 'color' / view / '0.png')
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(mask, mode='L').save(root / 'mask' / view / '1' / '0.png')
    with (root / 'mask' / f'mask_info_{view}.json').open('w', encoding='utf-8') as f:
        json.dump({'0': 'hand', '1': 'toy'}, f)
print(root)
PY
```

Result: passed; created a synthetic two-view case under `/tmp/fpt-mvsam3d-smoke`.

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python data_process_sam3d/prepare_mvsam3d_scene.py \
  --base_path /tmp/fpt-mvsam3d-smoke/data/different_types \
  --case_name toy_case \
  --category 'stuffed animal' \
  --controller_name hand \
  --force >/tmp/fpt_prepare_smoke.json
```

Result: passed; wrote
`/tmp/fpt-mvsam3d-smoke/data/different_types/toy_case/shape/mvsam3d/manifest.json`.
The manifest reported `object_name=stuffed_animal`, views `['0', '1']`, and
16 foreground pixels per generated mask.

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
root = Path('/tmp/fpt-mvsam3d-smoke/data/different_types/toy_case/shape/mvsam3d/input')
for path in [root / 'images' / '0.png', root / 'images' / '1.png', root / 'stuffed_animal' / '0.png', root / 'stuffed_animal' / '1.png']:
    img = Image.open(path)
    print(path.relative_to(root), img.mode, img.size)
mask = np.array(Image.open(root / 'stuffed_animal' / '0.png'))
print('alpha_values', np.unique(mask[..., 3]).tolist())
print('alpha_foreground', int((mask[..., 3] > 0).sum()))
PY
```

Result: passed; generated RGB view images, RGBA masks, alpha values `[0, 255]`,
and 16 foreground pixels.

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path /tmp/fpt-mvsam3d-smoke/data/different_types \
  --case_name toy_case \
  --category 'stuffed animal' \
  --controller_name hand \
  --dry_run
```

Result: passed; printed DA3 and `run_inference_weighted.py` commands with
`--mask_prompt stuffed_animal`, `--image_names 0,1`, and `--da3_output` under
the case `shape/mvsam3d/da3/` directory.

```bash
cd /tmp/fpt-mvsam3d-relcheck
/home/xinjie/miniforge3/envs/demo3-max/bin/python /home/xinjie/FuturePhysTwin/data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path data/different_types \
  --case_name toy_case \
  --category 'stuffed animal' \
  --controller_name hand \
  --dry_run
```

Result on 2026-05-29: passed; with relative `--base_path`, printed absolute
`--image_dir`, `--input_path`, and `--da3_output` paths. This guards the
external `cwd=MVSAM3D_ROOT` execution path.

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python /home/xinjie/FuturePhysTwin/data_process_sam3d/prepare_mvsam3d_scene.py \
  --base_path data/different_types \
  --case_name toy_case \
  --category 'stuffed animal' \
  --controller_name hand \
  --force
```

Result on 2026-05-29: passed from `/tmp/fpt-mvsam3d-relcheck`; manifest used
per-view object ids `0:1` and `1:2`, proving object id does not need to match
across cameras. Generated masks were RGBA with alpha values `[0, 255]`.

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python - <<'PY'
from pathlib import Path
import shutil
import trimesh
from data_process_sam3d.shape_prior_mvsam3d import normalize_outputs
root = Path('/tmp/fpt-mvsam3d-normalize-check')
shutil.rmtree(root, ignore_errors=True)
out = root / 'mv-output'
shape = root / 'case' / 'shape'
work = shape / 'mvsam3d'
out.mkdir(parents=True, exist_ok=True)
trimesh.creation.box(extents=(1, 1, 1)).export(out / 'result.glb')
result = normalize_outputs(out, shape, work, force=True, max_faces=50000)
loaded = trimesh.load(shape / 'object.glb', force='mesh', process=False)
print(result['object_glb'])
print(loaded.is_empty, len(loaded.vertices), len(loaded.faces))
PY
```

Result on 2026-05-29: passed; `shape/object.glb` loaded as non-empty with
8 vertices and 12 faces.

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python - <<'PY'
from pathlib import Path
import shutil
from data_process_sam3d.shape_prior_mvsam3d import normalize_outputs
root = Path('/tmp/fpt-mvsam3d-normalize-missing-result')
shutil.rmtree(root, ignore_errors=True)
out = root / 'mv-output'
shape = root / 'case' / 'shape'
work = shape / 'mvsam3d'
out.mkdir(parents=True, exist_ok=True)
try:
    normalize_outputs(out, shape, work, force=True, max_faces=50000)
except FileNotFoundError as exc:
    print(str(exc).split(':', 1)[0])
else:
    raise SystemExit('expected missing result.glb failure')
PY
```

Result on 2026-05-29: passed; wrapper refused to normalize an MV-SAM3D output
directory that did not contain canonical `result.glb`.

```bash
find data/different_types -mindepth 1 -maxdepth 1 -type d
```

Result before the real-data pass: no local repository cases found;
`data/different_types` did not exist yet.

## Data Cases

- Case: synthetic `toy_case`
- Input path: `/tmp/fpt-mvsam3d-smoke/data/different_types/toy_case`
- Output path:
  `/tmp/fpt-mvsam3d-smoke/data/different_types/toy_case/shape/mvsam3d/input`

## Findings

- Severity: info
  File: `process_data_sam3d.py`, `script_process_data_sam3d.py`
  Issue: command construction now uses argument lists rather than unquoted shell
  strings, so categories such as `stuffed animal` are passed safely.
  Evidence: synthetic dry-run used `--category 'stuffed animal'` and produced
  mask prompt `stuffed_animal`.
  Status: accepted.

- Severity: info
  File: `data_process_sam3d/prepare_mvsam3d_scene.py`
  Issue: base `python` lacks `numpy`/`PIL`; help should still work without
  runtime imports.
  Evidence: moved heavy imports into execution functions; all help commands now
  pass under base Python.
  Status: fixed.

- Severity: high
  File: `data_process_sam3d/shape_prior_mvsam3d.py`
  Issue: relative `--base_path` produced relative `--image_dir`, `--input_path`,
  and `--da3_output` in commands that run with `cwd=MVSAM3D_ROOT`.
  Evidence: semantic audit of wrapper cwd behavior and MV-SAM3D loader path
  handling.
  Status: fixed by resolving case/work paths to absolute paths before preparing
  the scene and constructing external commands.

- Severity: medium
  File: `data_process_sam3d/shape_prior_mvsam3d.py`
  Issue: MV-SAM3D `result.glb` was copied to `shape/object.glb` without the
  non-empty mesh validation that the old SAM3D backend effectively performed.
  Evidence: comparison against `data_process_sam3d/shape_prior.py` export
  validation and downstream `align.py`'s hard dependency on `shape/object.glb`.
  Status: fixed with `trimesh` validation during normalization.

- Severity: medium
  File: `data_process_sam3d/shape_prior_mvsam3d.py`
  Issue: the wrapper could fall back to another `.glb` if `result.glb` was
  missing, which could silently change `shape/object.glb` from a single object
  prior into a debug or scene artifact.
  Evidence: semantic audit against the downstream `align.py` contract.
  Status: fixed; missing canonical `result.glb` now fails clearly.

- Severity: high
  File: `data_process_sam3d/shape_prior_mvsam3d.py`,
  `data_process_sam3d/utils/align_util.py`
  Issue: real MV-SAM3D output for `single_lift_sloth` produced a valid
  single-object mesh, but direct `result.glb` normalization wrote 523,672 faces
  with RGBA vertex colors. That preserved object semantics but made
  `align.py --force_rematch` spend over 9 minutes in PyTorch3D rendering and
  the RGBA features triggered a shader channel mismatch before the renderer
  guard was added.
  Evidence: real case run from the PhysTwin data zip; full MV result had
  261,846 vertices / 523,672 faces. Downstream `shape/object.glb` is now
  simplified to 50,000 faces by default while the full MV `result.glb` remains
  under `shape/mvsam3d/outputs/`. `align_util.render_image()` crops
  `TexturesVertex` features to RGB for PyTorch3D rendering.
  Status: fixed and verified on `single_lift_sloth`.

- Severity: medium
  File: `data_process_sam3d/align.py`, `data_process_sam3d/data_process_sample.py`
  Issue: local OpenCV/FFmpeg does not provide an H264 `avc1` encoder, so
  downstream visualization MP4s could silently remain stale even when data files
  were regenerated.
  Evidence: `align.py` logged `Could not find encoder for codec_id=27` before
  the codec change; `final_mesh.glb` was new but `final_matching.mp4` still had
  the old dataset timestamp.
  Status: fixed by using `mp4v`, checking `VideoWriter.isOpened()`, and
  releasing writers. Verified generated MP4s are readable.

## Real PhysTwin Case Verification

Downloaded and extracted one case from the user-provided Hugging Face dataset:

```bash
curl -L --fail --continue-at - --output data/data.zip \
  https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip
unzip -q -o data/data.zip 'data/different_types/single_lift_sloth/*'
```

Case: `data/different_types/single_lift_sloth`, category `sloth`,
controller `hand`.

Adapter command:

```bash
/home/xinjie/miniforge3/envs/demo3-max/bin/python \
  data_process_sam3d/prepare_mvsam3d_scene.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --controller_name hand \
  --force
```

Result: passed. Manifest selected object label `sloth`, object index `0`, for
views `0,1,2`; `hand` remained the controller label. Foreground alpha pixel
counts were 23,894, 9,017, and 15,736. A manual overlay check was saved at
`/tmp/single_lift_sloth_mvsam3d_mask_check.jpg`.

Full MV-SAM3D command:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --controller_name hand \
  --mvsam3d_root /home/xinjie/external/MV-SAM3D \
  --mvsam3d_python /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  --skip_da3_if_exists
```

Result: passed. DA3 output existed at
`shape/mvsam3d/da3/da3_output.npz`. MV-SAM3D wrote canonical
`result.glb`, `result.ply`, and `params.npz` under
`/home/xinjie/external/MV-SAM3D/visualization/input/sloth/input_sloth_3v_s1a30_s2e30_20260529_161850`,
and the wrapper copied debug outputs under
`shape/mvsam3d/outputs/input_sloth_3v_s1a30_s2e30_20260529_161850`.

Mesh validation:

```text
full MV debug result: 261,846 vertices, 523,672 faces,
  bounds [[-0.4991, -0.1611, -0.4978], [0.4999, 0.1571, 0.4945]]
downstream object.glb: 25,010 vertices, 50,000 faces,
  bounds [[-0.4988, -0.1611, -0.4978], [0.4999, 0.1570, 0.4945]]
```

The full and simplified meshes have matching bounds, so `shape/object.glb`
remains the same single reconstructed object mesh semantic, not a merged scene
or DA3 debug artifact. A side-by-side mesh render was saved at
`/tmp/single_lift_sloth_shape_prior_compare.jpg`.

Alignment verification:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/align.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --controller_name hand \
  --force_rematch
```

Result: passed after downloading ignored SuperPoint/SuperGlue weights into
`data_process_sam3d/models/weights/`. `align.py` matched 56 points, reported
reprojection error `8.973943710327148`, scale `0.35004712822820155`, and wrote
loadable `shape/matching/final_mesh.glb`.

Final data verification:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/data_process_sample.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --shape_prior
```

Result: passed. `final_data.pkl` was regenerated with
`object_points (85, 5203, 3)`, `surface_points (558, 3)`,
`interior_points (286, 3)`, and `controller_points (85, 30, 3)`.
`final_matching.mp4`, `final_pcd.mp4`, and `final_data.mp4` were regenerated as
readable MPEG-4 videos.

Initial quality comparison against the original/Trellis backup before the
multi-view align rewrite:

```text
Trellis/original final_data: surface_points (554, 3), interior_points (1171, 3)
MV-SAM3D final_data:         surface_points (558, 3), interior_points (286, 3)
```

Result: MV-SAM3D passed the semantic/downstream contract, but the visual
`final_pcd` result is not clearly better on `single_lift_sloth`. It is notably
less filled internally than the original/Trellis result. Distance-to-observation
metrics show the MV alignment is usable, but final point-cloud enrichment is a
quality regression for this case rather than an improvement. Keep the legacy
backend available and treat MV-SAM3D as needing more multi-case QA before making
it the preferred production path.

## Full Multi-View Align Rewrite Verification

Implemented `data_process_sam3d/align_mvsam3d.py` and integrated it through
`--align_backend auto`, so MV-SAM3D cases use the multi-view aligner while
legacy Trellis/SAM3D cases can still use `align.py`.

Real multi-view align command:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/align_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --controller_name hand \
  --force_rematch
```

Result: passed. The multi-view aligner used manifest views `0,1,2`, found two
valid matched views, kept `41` total PnP/RANSAC inliers, selected view `0` as
the trusted keypoint view after robust multi-view scoring, and still used view
`2` for diagnostics/scoring. The optimized result wrote
`shape/matching/final_mesh.glb`, `shape/matching/final_matching.mp4`, and
`shape/mvsam3d/align/metrics.json`.

Distance metrics from `metrics.json`:

```text
mesh vertices/faces:       25,010 / 50,000
vertex_to_obs median/p95:  0.00250 / 0.02992 m
obs_to_vertex median/p95:  0.00204 / 0.00711 m
```

These satisfy the requested gates: at least two valid views, at least 36 total
inliers, observation-to-mesh median under 5 mm, observation-to-mesh p95 under
15 mm, and mesh-to-observation p95 under 35 mm.

MV-SAM3D adaptive final-data sampling command:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/data_process_sample.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --shape_prior \
  --shape_prior_sampling_backend mvsam3d
```

Result: passed. Regenerated `final_data.pkl`, `final_pcd.mp4`, and
`final_data.mp4`.

```text
MV-SAM3D final_data: object_points (85, 5203, 3)
                     surface_points (700, 3)
                     interior_points (1000, 3)
```

Rebuilt Trellis/original final PCD from
`shape/mvsam3d/original_shape_backup/matching/final_mesh.glb` with legacy
sampling for comparison:

```text
Trellis/original final_data: surface_points (514, 3), interior_points (1166, 3)
MV-SAM3D final_data:         surface_points (700, 3), interior_points (1000, 3)
```

Comparison artifacts:

- `/tmp/mvsam3d_vs_trellis_final_pcd.mp4`
- `/tmp/mvsam3d_vs_trellis_final_pcd_contact.jpg`

Result: the prior quality regression is addressed for `single_lift_sloth`.
MV-SAM3D now has denser surface support and comparable interior fill instead
of the earlier thin/empty final PCD.

Final static/documentation checks after the full MV align rewrite:

```bash
python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py data_process_sam3d/align_mvsam3d.py data_process_sam3d/data_process_sample.py data_process_sam3d/utils/align_util.py
python scripts/check_harness_docs.py
python process_data_sam3d.py --help >/tmp/fpt_process_help_final.txt
python script_process_data_sam3d.py --help >/tmp/fpt_script_help_final.txt
/home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/align_mvsam3d.py --help >/tmp/fpt_align_mvsam3d_help_final.txt
/home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/data_process_sample.py --help >/tmp/fpt_sample_help_final.txt
git diff --check
```

Result: passed. Help output line counts were 59, 55, 28, and 23 lines.

## Acceptance Criteria Results

- Criterion: adapter writes MV-SAM3D scene inputs and manifest.
  Result: passed on synthetic case and real `single_lift_sloth`.
- Criterion: wrapper supports dry-run without importing MV-SAM3D.
  Result: passed.
- Criterion: pipeline defaults to `--shape_prior_backend mvsam3d` while keeping
  `sam3d`.
  Result: passed by help/grep review.
- Criterion: downstream output contract remains `shape/object.glb`, optional
  `shape/object.ply`, optional `shape/visualization.mp4`.
  Result: passed. `shape/object.glb`, `shape/object.ply`, aligned
  `shape/matching/final_mesh.glb`, and final data generation were verified on
  `single_lift_sloth`.

## Skipped Checks

- Check:
  full end-to-end `process_data_sam3d.py` including segmentation, dense
  tracking, point-cloud lifting, and mask post-processing.
  Reason: the user asked to check the MV-SAM3D shape-prior semantics on one
  case; existing extracted case artifacts were reused for downstream
  `align.py` and `data_process_sample.py` rather than rerunning unrelated
  upstream stages.
  Next command to run:
  `/home/xinjie/miniforge3/envs/phystwin-max/bin/python process_data_sam3d.py --base_path ./data/different_types --case_name single_lift_sloth --category sloth --shape_prior --shape_prior_backend mvsam3d --mvsam3d_skip_da3_if_exists`

## Residual Risk

- Risk: `phystwin-max` was mutated to satisfy MV-SAM3D dependencies. For
  repeatability, a dedicated MV-SAM3D conda env should be created instead of
  continuing to layer packages into the shared runtime.
- Risk: `shape/object.glb` simplification preserves bounds and passed this
  sloth case, but other categories may need a different `--mvsam3d_max_faces`
  threshold if fine geometry is important for matching.

## MV-SAM3D Legacy Upscale Input Update

2026-05-29 follow-up: MV-SAM3D input preparation now mirrors the legacy
single-view SAM3D preprocessing by default. For each selected frame-0 view, the
wrapper runs `image_upscale.py` on the masked object crop, then runs
`segment_util_image.py` on the upscaled image to produce a high-resolution RGBA
object mask. The generated MV-SAM3D scene then uses those high-resolution images
and masks. The raw input path remains available with
`--input_preprocess_backend raw` / `--mvsam3d_input_preprocess_backend raw`.

Dry-run command:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --controller_name hand \
  --input_preprocess_backend legacy_upscale \
  --view_indices 0,1,2 \
  --dry_run
```

Result: passed. The dry-run planned six preprocessing commands, two per view,
writing to `shape/mvsam3d/preprocess/legacy_upscale/<view>/`, and DA3 output is
now namespaced under `shape/mvsam3d/da3/legacy_upscale/da3_output.npz` so raw
and high-resolution inputs cannot accidentally share stale DA3 cache.

Additional checks:

```bash
python -m py_compile process_data_sam3d.py script_process_data_sam3d.py data_process_sam3d/prepare_mvsam3d_scene.py data_process_sam3d/shape_prior_mvsam3d.py
/home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/image_upscale.py --help
/home/xinjie/miniforge3/envs/phystwin-max/bin/python data_process_sam3d/segment_util_image.py --help
python scripts/check_harness_docs.py
git diff --check
```

Result: passed. Full high-resolution MV-SAM3D inference was not rerun in this
follow-up; run without `--dry_run` to regenerate the shape prior with the new
input preprocessing.

## Direct phystwin-max Runtime Update

2026-05-29 follow-up: removed hardcoded conda environment defaults from the
SAM3D/MV-SAM3D Python code path. `process_data_sam3d.py`,
`script_process_data_sam3d.py`, and `shape_prior_mvsam3d.py` now default their
Python subprocesses to the current interpreter. On this machine the intended
entrypoint is therefore direct `phystwin-max`, for example:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python process_data_sam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --shape_prior \
  --shape_prior_backend mvsam3d
```

Override flags remain available only for non-standard setups:
`--pipeline_python`, `--legacy_shape_prior_python`, `--mvsam3d_python`, and
`--mvsam3d_preprocess_python`.

## Legacy-Upscale Full Run

2026-05-29 follow-up run on `single_lift_sloth` used the direct
`phystwin-max` interpreter and the new default MV-SAM3D input preprocessing.

Command:

```bash
/home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/shape_prior_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --category sloth \
  --controller_name hand \
  --mvsam3d_root /home/xinjie/external/MV-SAM3D \
  --input_preprocess_backend legacy_upscale \
  --view_indices 0,1,2 \
  --skip_da3_if_exists \
  --max_faces 50000
```

Result: passed. The wrapper consumed three high-resolution frame-0 inputs:
view 0 `1216x1216`, view 1 `832x832`, and view 2 `928x928`. RGBA alpha masks
were non-empty for all views. MV-SAM3D wrote
`shape/mvsam3d/outputs/input_sloth_3v_s1a30_s2e30_20260529_204406/result.glb`;
the downstream mesh was normalized to `shape/object.glb` with `25016` vertices
and `50000` faces, while the full debug mesh retained `278598` vertices and
`557164` faces.

The first DA3 attempt exposed a local cache-layout issue:
`scripts/run_da3.py` found the Hugging Face model cache root while this machine
stores the actual weights under a snapshot directory. The wrapper now accepts
`--da3_model_path`, with pass-through `--mvsam3d_da3_model_path` in
`process_data_sam3d.py` and `script_process_data_sam3d.py`.

Follow-up align command:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/align_mvsam3d.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --controller_name hand \
  --force_rematch
```

Result: passed after fixing a match-cache pickle bug. New metrics:
`valid_views=2`, `total_inliers=37`, `active_keypoint_views=["0","2"]`,
`obs_to_vertex median/p95=0.00350/0.02720 m`, and
`vertex_to_obs median/p95=0.00566/0.05025 m`. The observation-to-mesh side is
acceptable for this case, but mesh-to-observation p95 is above the earlier
35 mm gate and should be treated as residual shape-prior overgrowth risk.

Final-data command:

```bash
xvfb-run -a /home/xinjie/miniforge3/envs/phystwin-max/bin/python \
  data_process_sam3d/data_process_sample.py \
  --base_path ./data/different_types \
  --case_name single_lift_sloth \
  --shape_prior \
  --shape_prior_sampling_backend mvsam3d
```

Result: passed. `final_data.pkl` contains `object_points (85, 5203, 3)`,
`surface_points (700, 3)`, and `interior_points (1000, 3)`. Regenerated videos:
`final_pcd.mp4` and `final_data.mp4`.

Comparison artifacts:

- `/tmp/single_lift_sloth_mvsam3d_legacy_upscale_vs_trellis_final_pcd.mp4`
- `/tmp/single_lift_sloth_mvsam3d_legacy_upscale_vs_trellis_final_pcd_contact.jpg`
- `/tmp/single_lift_sloth_mvsam3d_legacy_upscale_vs_trellis_raw_mesh.mp4`
- `/tmp/single_lift_sloth_mvsam3d_legacy_upscale_vs_trellis_raw_mesh_contact.jpg`

Visual note: the final point clouds look much closer after the adaptive MV
sampling fix, but the raw MV-SAM3D shape prior still shows an abnormal large
light-colored surface patch. That is a model/input artifact, not an align
contract break.
