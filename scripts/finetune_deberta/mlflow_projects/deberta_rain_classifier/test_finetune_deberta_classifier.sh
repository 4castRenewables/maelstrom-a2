poetry run python finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_train.nc \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_validate.nc \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_test.nc \
    --model_path ../../../../models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --debug

