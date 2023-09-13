# rsync -Pvra ../../../data/tweets/2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc /tmp/
additional_args="$@"
poetry run python ../../relevance_classifier/build_dataset_rain_classifier.py \
    --filename_dataset_to_split /tmp/2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc \
    --output_dir /tmp/relevance_classifier \
    --task_name dataset_rain_split \
    --test_size 0.2 \
    --validation_size 0.2 \
    --key_stratify raining \
    --classifier_domain rain \
    --key_input text \
    --key_rain tp_h \
    --threshold_rain 7e-6 \
    --key_relevance prediction_relevant \
    --select_relevant False\
    --debug \
    ${additional_args}
