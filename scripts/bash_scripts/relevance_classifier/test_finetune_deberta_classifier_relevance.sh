additional_args="$@"
source ../env_dev.sh
poetry run python ../../relevance_classifier/finetune_deberta_classifier.py \
    --model_path /home/kristian/Projects/a2/models/deberta-v3-small \
    --filename_dataset_train /tmp/relevance_classifier/dataset_split/RelevanceClassifierDataset2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_train.nc \
    --filename_dataset_validate /tmp/relevance_classifier/dataset_split/RelevanceClassifierDataset2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_validate.nc \
    --filename_dataset_test /tmp/relevance_classifier/dataset_split/RelevanceClassifierDataset2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_test.nc \
    -esteps 1 \
    --output_dir /tmp/test_relevance_classifier/ \
    --task_name relevance_classification \
    --mlflow_experiment_name maelstrom-a2-relevance \
    --classifier_domain relevance \
    --key_output relevant \
    --debug \
    ${additional_args}