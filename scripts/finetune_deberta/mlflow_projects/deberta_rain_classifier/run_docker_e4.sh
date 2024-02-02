docker_image_id=$(docker run -t -d --gpus all --ipc=host -e HOME=$HOME -e USER=$USER \
    --mount "type=bind,source=${jube_benchmark_home}/../../scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/,target=/scripts" \
--mount "type=bind,source=${base_path_dataset},target=${base_path_dataset} nvcr.io/nvidia/pytorch:24.01-py3")
export docker_image_id=${docker_image_id}