poetry run python build_dataset_rain_classifier.py \
    --filename_tweets /home/kristian/Projects/a2/data/bootcamp2023/tweets/tweets_2017_era5_normed_filtered.csv \
    --output_dir /tmp/dataset_rain_classifier/ \
    --key_precipitation_station tp_mm_station \
    --dataset_backend pandas \
    --ignore_tracking