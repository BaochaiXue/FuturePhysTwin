: <<'DOC'
gs_run_simulate.sh

Inputs
------
- Canonical Gaussian checkpoints in ``gaussian_output/<scene>/<exp_name>/``.
- Scene assets in ``data/gaussian_data/<scene>/`` containing evaluation camera paths.

Outputs
-------
- Dynamic render frames in ``gaussian_output_dynamic/<scene>/<view>/``.
- Corresponding MP4 previews ``gaussian_output_dynamic/<scene>/<view>.mp4``.
DOC

output_dir="./gaussian_output_dynamic"
mkdir -p "${output_dir}"

views=("0" "1" "2")
#views=("0")

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

for scene_path in ./data/gaussian_data/*/; do
    if [[ ! -d "${scene_path}" ]]; then
        continue
    fi
    scene_name="$(basename "${scene_path%/}")"
    python gs_render_dynamics.py \
        -s "${scene_path%/}" \
        -m ./gaussian_output/${scene_name}/${exp_name} \
        --name ${scene_name} \

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done
