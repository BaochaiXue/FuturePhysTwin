# Verification Matrix

PhysTwin verification often depends on large data, GPU availability, CUDA
extensions, and long-running training. The harness uses layered verification so
agents can provide useful evidence even when full runs are unavailable.

## Levels

### Level 0: Static

Use for docs, scripts, refactors, and import-safe changes.

- Syntax checks such as `python -m py_compile <files>`.
- Shell syntax checks such as `bash -n <script>`.
- Harness docs check: `python scripts/check_harness_docs.py`.
- Focused grep/read checks for renamed flags or paths.

### Level 1: Import and Dry Run

Use when a module can be imported without requiring full datasets.

- Import changed modules.
- Run help output for changed CLI scripts.
- Execute no-op or tiny-case paths if supported.
- Validate config parsing.

### Level 2: Small Case Functional

Use when a small local case exists.

- Run the relevant script against one case.
- Verify expected output files and shapes.
- Save output paths in the QA report.
- Compare representative visualizations when task changes rendering or masks.

### Level 3: Full Pipeline

Use for behavior that affects reconstruction quality, simulation, rendering,
or evaluation metrics.

- Process raw data or documented processed data.
- Run optimization/training/inference as required.
- Run rendering/evaluation commands.
- Record metrics, visual artifacts, GPU, CUDA, and runtime.

## Domain Checks

### Data Processing

- Confirm expected inputs: `color`, `depth`, `calibrate.pkl`, `metadata.json`.
- Check masks, tracking files, point clouds, and shape priors are created in the
  expected case directory.
- Probe missing or partial data behavior.

### Gaussian Rendering

- Verify RGB, alpha/segmentation, depth, and normal tensor shapes.
- Confirm background and mask handling is explicit.
- Save visual comparisons for changes to rendering, compositing, or evaluation.

### Simulation and Training

- Record config, case name, seed if available, iteration count, and device.
- Verify checkpoints or output folders are written.
- Watch for silent CPU/GPU placement changes and memory regressions.

### Evaluation

- Record metric command, input folders, and output file paths.
- Avoid claiming quality improvements without metric or visual evidence.

### Interactive Playground

- Verify launch command and case name.
- If manual interaction is needed, record controls tested and observed behavior.
- For remote control changes, test both physical and virtual input paths when
  possible.

## Skipped Checks

Skipped checks are acceptable only when the QA report states:

- The exact check skipped.
- Why it could not run.
- What lower-level evidence was gathered instead.
- What command should be run next when resources are available.
