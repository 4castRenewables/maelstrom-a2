MODEL_NAME="electra_base"
python ../../../scripts/relevance_classifier/build_dataset_relevance_classifier.py \
       --output_dir /p/project/deepacf/maelstrom/ehlert1/data/dataset_split/ \
       --task_name 2017-2020_{MODEL_NAME} \
       --filename_tweets_with_keywords 2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc \
       --data_filename_irrelevant tweets_no_keywords_2020-02-13T00:00:00.000Z_2020-02-14T00:00:00_locations_bba_era5.nc \
       --tweets_dir /p/scratch/deepacf/maelstrom/maelstrom_data/ap2/data/tweets/ \
       --n_tweets_irrelevant 500000 \
       --key_rain tp_h \
       --threshold_rain 1e-4 \
       --test_size 0.2 \
       --validation_size 0.2

