poetry run python ../finetune_deberta.py \
    --run_folder /tmp/test_run/ \
    --model_path /home/kristian/Projects/a2/models/deberta-v3-small \
    --data_filename tweets_2017_01_era5_normed_filtered_predicted_simpledeberta.nc \
    -esteps 1 \
    -ts 0.99