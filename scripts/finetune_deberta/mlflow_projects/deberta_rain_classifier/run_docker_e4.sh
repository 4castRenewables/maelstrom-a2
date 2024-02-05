available_images=$(docker ps -af name=ap2 -q)
echo available_images: ${available_images}
if [ -n "$available_images" ]; then
    docker stop ${available_images}
    docker rm ${available_images}
fi
# if [ -z "$image_exists" ]; then
docker run --name ap2_maelstrom-${SLURM_JOB_ID} -t -d --gpus all --ipc=host -e HOME=$HOME -e USER=$USER \
        --mount "type=bind,source=${jube_benchmark_home}/../../scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/,target=/scripts" \
        --mount "type=bind,source=${base_path_dataset},target=${base_path_dataset}" \
        --mount "type=bind,source=${model_base_path},target=${model_base_path}" \
        --mount "type=bind,source=${SLURM_SUBMIT_DIR},target=${SLURM_SUBMIT_DIR}" \
        --env SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}
        kristian4cast/ml:a2-cuda
# fi
