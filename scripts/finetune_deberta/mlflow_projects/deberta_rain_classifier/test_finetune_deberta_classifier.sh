poetry run python finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_train.nc \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_validate.nc \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_test.nc \
    --model_path ../../../../models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --trainer_name deep500 \
    --debug \
    --ignore_tracking

