MODEL_NAME="electra_base"
python ../../../scripts/relevance_classifier/build_dataset_rain_classifier.py \
    --filename_dataset_to_split /p/project/deepacf/maelstrom/ehlert1/data/dataset_split/2017-2020//relevance_prediction//RainingClassifierDataset2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar_relevant_predicted.nc \
    --output_dir /p/project/deepacf/maelstrom/ehlert1/data/dataset_split/ \
    --task_name 2017-2020_${MODEL_NAME} \
    --test_size 0.2 \
    --validation_size 0.2 \
    --key_stratify raining \
    --classifier_domain rain \
    --key_output raining \
    --key_input text \
    --key_rain tp_h \
    --threshold_rain 1e-4 \
    --key_relevance prediction_relevant \
    --select_relevant True
