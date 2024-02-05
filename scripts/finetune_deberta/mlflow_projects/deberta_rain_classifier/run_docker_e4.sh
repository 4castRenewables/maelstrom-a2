image_exists=$(docker ps | grep ap2_maelstrom)
echo $image_exists
if [ -z "$image_exists" ]; then
    docker run --name ap2_maelstrom -t -d --gpus all --ipc=host -e HOME=$HOME -e USER=$USER \
        --mount "type=bind,source=${jube_benchmark_home}/../../scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/,target=/scripts" \
        --mount "type=bind,source=${base_path_dataset},target=${base_path_dataset}" kristian4cast/ml:a2-cuda
fi
