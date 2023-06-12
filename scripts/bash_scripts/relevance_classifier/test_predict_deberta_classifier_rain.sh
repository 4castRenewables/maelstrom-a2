additional_args="$@"
source env_dev.sh
poetry run python ../predict_deberta_classifier.py \
    --filename_dataset_predict /tmp/relevance_classifier/dataset_split/WeatherStationDataset2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc \
    --path_raw_model /home/kristian/Projects/a2/models/deberta-v3-small \
    --path_trained_model /tmp/test_relevance_classifier//relevance_classification/checkpoint-4 \
    --output_dir /tmp/test_prediction_relevance/ \
    --task_name relevance_classification \
    --mlflow_experiment_name maelstrom-a2-rain \
    --classifier_domain rain \
    --key_output raining_station \
    --debug \
    ${additional_args}
