#!/bin/bash
. ~/.bashrc
. /p/home/jusers/ehlert1/juwels/env.sh
conda init
conda activate a2-benchmarking
python ../scripts/finetune_bert.py --run_folder test/ --data_filename tweets_2017_01_era5_normed_filtered_predicted_simpledeberta.nc
