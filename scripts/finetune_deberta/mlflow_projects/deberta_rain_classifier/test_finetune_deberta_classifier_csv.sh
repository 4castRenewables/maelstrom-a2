poetry run python finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_train.csv \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_validate.csv \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//tweets_2017_era5_normed_filtered_test.csv \
    --model_path ../../../../models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --trainer_name deep500 \
    --dataset_backend pandas \
    --debug \
    --ignore_tracking

