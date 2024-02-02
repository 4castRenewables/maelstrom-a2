poetry run python evaluate_deberta_classifier.py \
    --filename_dataset_evaluate /tmp/dataset_rain_classifier//dataset_split_thresh6M3//Weather_stations_2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc \
    --model_trained_path /tmp/trained_model//dataset_split_thresh6M3/save_model/checkpoint-4/ \
    --output_dir /tmp/evaluate_model/ \
    --debug

