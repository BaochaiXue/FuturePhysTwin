: <<'DOC'
evaluate.sh

Inputs
------
- Predictions in ``experiments/<case>/`` (mesh/trajectory exports).
- Evaluation assets in ``data/different_types/<case>/`` and ``data/render_eval_data/<case>/``.
- Render sequences in ``gaussian_output_dynamic``.

Outputs
-------
- Aggregated quantitative metrics written to ``results/`` (CSV files, logs).
DOC

python evaluate_chamfer.py
python evaluate_track.py
python gaussian_splatting/evaluate_render.py
