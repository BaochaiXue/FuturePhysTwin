: <<'DOC'
gs_run_simulate_white.sh

Inputs
------
- Colour-stage checkpoints in ``gaussian_output/<scene>/<exp_name>/``.
- Scene definitions within ``data/gaussian_data/<scene>/``.

Outputs
-------
- White-background render frames ``gaussian_output_dynamic_white/<scene>/<view>/``.
- MP4 previews ``gaussian_output_dynamic_white/<scene>/<view>.mp4``.
DOC

output_dir="./gaussian_output_dynamic_white"
mkdir -p "${output_dir}"

views=("0" "1" "2")
# views=("0")

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
        --white_background \
        --output_dir ${output_dir}

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
        if [[ -d "${output_dir}/${scene_name}/${view_name}_black" ]]; then
            python gaussian_splatting/img2video.py \
                --image_folder ${output_dir}/${scene_name}/${view_name}_black \
                --video_path ${output_dir}/${scene_name}/${view_name}_black.mp4
        fi
    done

done
