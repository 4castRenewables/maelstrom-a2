import argparse
import pathlib

import a2.training
import a2.utils


def get_parser(formatter_class=argparse.ArgumentDefaultsHelpFormatter):
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    return parser


def dataset_tweets(parser):
    parser.add_argument(
        "--tweets_dir",
        "-inpath",
        type=str,
        default=a2.utils.file_handling.get_folder_data() / "tweets/",
        help="Directory where input netCDF-files are stored.",
    )


def dataset(parser):
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar",  # noqa: E501
        help="Dataset prefix determined during split of dataset.",
    )
    parser.add_argument(
        "--data_description",
        "-datad",
        type=str,
        default="tweets 2017-2020, keywords emojis as description, keywords only",
        help="Data description purely used for logging.",
    )


def evaluation(parser):
    parser.add_argument(
        "--evaluation_strategy", "-estrat", type=str, default="epoch", help="When to evaluate steps/epoch/no"
    )
    parser.add_argument(
        "--eval_steps",
        "-esteps",
        type=int,
        default=None,
        help="Number of steps between evaluations if evaluation strategy is set to steps.",
    )


def classifier(parser):
    parser.add_argument("--key_text", type=str, default="text_normalized", help="Key for text in input file.")
    parser.add_argument(
        "--classifier_domain", choices=["rain", "relevance"], type=str, required=True, help="Domain of classifier"
    )
    parser.add_argument(
        "--key_input",
        type=str,
        default="text",
        choices=["text"],
        help="Column name of dataset that corresponds to input of classifier.",
    )
    parser.add_argument(
        "--key_output",
        type=str,
        default="raining",
        choices=["raining", "relevant", "raining_station"],
        help="Column name of dataset that corresponds to output of classifier.",
    )


def dataset_training(parser):
    parser.add_argument(
        "--filename_dataset_train", required=True, type=str, help="Filename of dataset used for training."
    )
    parser.add_argument(
        "--filename_dataset_validate", required=True, type=str, help="Filename of dataset used for validation."
    )
    parser.add_argument(
        "--filename_dataset_test", required=True, type=str, help="Filename of dataset used for testing."
    )


def output(parser):
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Folder to save results.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of task, results will be saved in this subdirectory of output.",
    )


def dataset_rain(parser):
    parser.add_argument("--key_rain", type=str, default="tp_h", help="Key for rain data in input file.")
    parser.add_argument(
        "--threshold_rain",
        type=float,
        default=1e-8,
        help="Threshold above which precipitation is considered raining in input file.",
    )


def dataset_weather_stations(parser):
    parser.add_argument(
        "--kms_within_station",
        "-kms",
        type=float,
        default=1,
        help="Distance to weather station. Used to create weather station dataset.",
    )
    parser.add_argument(
        "--key_distance_weather_station",
        type=str,
        default="station_distance_km",
        help="Key in dataset that shows distance to weather station. Used to create weather station dataset.",
    )
    parser.add_argument(
        "--weather_station_dataset_prefix",
        type=str,
        default="WeatherStationDataset",
        help="Prefix of weather station dataset.",
    )


def dataset_relevance(parser):
    parser.add_argument(
        "--key_relevance",
        type=str,
        default="relevant",
        help="Variable name that specifies relevance of Tweet.",
    )


def dataset_relevance_split(parser):
    parser.add_argument(
        "--filename_tweets_with_keywords",
        type=str,
        default="2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered.nc",
        help="Filename of training data.",
    )
    parser.add_argument(
        "--data_filename_irrelevant",
        type=str,
        default="2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered.nc",
        help="Filename of training data.",
    )
    parser.add_argument(
        "--n_tweets_irrelevant",
        type=int,
        default=-1,
        help="Number of irrelevant tweets used to create dataset, "
        "use all irrelevant tweets by default (`n_tweets_irrelevant=-1`).",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="split_relevance",
        help="Folder to save split datasets.",
    )
    parser.add_argument(
        "--raining_classifier_dataset_prefix",
        type=str,
        default="RainingClassifierDataset",
        help="Prefix of raining classifier dataset.",
    )
    parser.add_argument(
        "--relevance_classifier_dataset_prefix",
        type=str,
        default="RelevanceClassifierDataset",
        help="Prefix of relevance classifier dataset.",
    )


def dataset_split_sizes(parser):
    parser.add_argument("--test_size", "-ts", type=float, default=0.2, help="Fraction of test set.")
    parser.add_argument("--validation_size", "-vs", type=float, default=0.2, help="Fraction of validation set.")


def dataset_split(parser):
    dataset_split_sizes(parser)
    parser.add_argument(
        "--filename_dataset_to_split",
        required=True,
        type=pathlib.Path,
        help="Filename of dataset that will be split.",
    )
    parser.add_argument(
        "--key_stratify",
        type=str,
        default=None,
        help="Key used to stratify dataset split.",
    )


def dataset_select(parser):
    parser.add_argument(
        "--select_relevant",
        type=custom_boolean,
        default=True,
        help="Whether to only select Tweets as revelant (`args.key_relevance`==True).",
    )


def predict_model(parser):
    parser.add_argument(
        "--filename_dataset_predict",
        required=True,
        type=pathlib.Path,
        help="Filename of dataset that predictions will be made for.",
    )
    parser.add_argument(
        "--path_raw_model",
        type=str,
        required=True,
        help="Directory where untrained model saved.",
    )
    parser.add_argument(
        "--path_trained_model",
        type=str,
        required=True,
        help="Directory where trained model saved.",
    )


def model(parser):
    parser.add_argument(
        "--model_path",
        "-in",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/deberta-v3-small/",
        help="Directory where input netCDF-files are stored.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=a2.training.model_configs.SUPPORTED_MODELS,
        default="deberta_small",
        help="Name of model, sets default hyper parameters.",
    )
    parser.add_argument(
        "--output_dir_model",
        "-outdir",
        type=str,
        default=a2.utils.file_handling.get_folder_models(),
        help="Output directory where model is saved.",
    )
    parser.add_argument(
        "--trainer_name",
        type=str,
        default="default",
        choices=a2.training.model_configs.SUPPORTED_TRAINERS,
        help="Trainer class selected by its name.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="default_loss",
        choices=a2.training.model_configs.SUPPORTED_LOSSES,
        help="Loss used for training selected by its name.",
    )


def model_classifier(parser):
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels in classification task.",
    )


def run(parser):
    parser.add_argument(
        "--run_folder", type=str, required=True, help="Output folder where model is saved in `output_dir`."
    )


def mlflow(parser):
    figures(parser)
    parser.add_argument("--run_name", type=str, default="RUNNAME", help="Name of run used for logging only.")
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="maelstrom-a2-train",
        help="Name MLflow experiment where results are logged.",
    )


def figures(parser):
    parser.add_argument(
        "--figure_folder", type=str, default="figures/", help="Directory where input netCDF-files are stored."
    )


def hyperparameter_basic(parser):
    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed value.")
    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")


def hyperparameter(parser):
    hyperparameter_basic(parser)
    # arg name should match name of hyper parameter used in `model_configs.py`!
    parser.add_argument("--epochs", type=int, default=1, help="Numer of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per mini-batch.")
    parser.add_argument("--learning_rate", type=float, default=3e-05, help="Learning rate to train model.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay rate to train model.")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio to train model.")
    parser.add_argument("--warmup_steps", type=float, default=500, help="Warmup steps to train model.")
    parser.add_argument(
        "--hidden_dropout_prob", "-hdp", type=float, default=0.1, help="Probability of hidden droup out layer."
    )
    parser.add_argument("--cls_dropout", "-cd", type=float, default=0.1, help="Dropout probability in classifier head.")
    parser.add_argument("--lr_scheduler_type", "-lst", type=str, default="linear", help="Learning rate scheduler type.")

    parser.add_argument(
        "--base_model_weights_fixed",
        "-weightsfixed",
        action="store_true",
        help="Weights of base model (without classification head) are fixed (not trainable).",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Steps after which model is saved.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Steps after which model logs are written.")


def benchmarks(parser):
    tracking(parser)
    parser.add_argument(
        "--log_gpu_memory",
        action="store_true",
        help="Monitor Cuda memory usage.",
    )


def tracking(parser):
    parser.add_argument(
        "--ignore_tracking",
        action="store_true",
        help="Do not use mantik tracking.",
    )


def debug(parser):
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to toggle debug mode.",
    )


def custom_boolean(boolean: str) -> bool:
    """Custom argparse type for booleans"""
    if boolean == "False" or boolean == "false" or boolean == "0":
        return False
    elif boolean == "True" or boolean == "true" or boolean == "1":
        return True
    else:
        raise argparse.ArgumentTypeError(f"Couldn't convert {boolean=} to boolean!")
