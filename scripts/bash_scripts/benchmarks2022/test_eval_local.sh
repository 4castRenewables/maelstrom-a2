poetry run python ../evaluate_deberta.py \
    --test_size 0.02 \
    --model_path /home/kristian/Projects/a2/models/model_weights/output_rainprediction_simpledeberta/checkpoint-816 \
    --output_dir /tmp/test_eval/ \
    --data_filename tweets_2017_01_era5_normed_filtered_predicted_simpledeberta.nc