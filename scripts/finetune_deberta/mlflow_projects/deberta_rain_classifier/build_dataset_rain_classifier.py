import logging
import os

import a2.dataset.load_dataset
import a2.dataset.utils_dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.training_hugging
import a2.utils.argparse
import utils_scripts

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main(args):
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    """Split labeled Tweets into train, test, validation set"""
    logger.info(f"Args used: {args.__dict__}")
    path_output = utils_scripts._determine_path_output(args)
    path_figures = os.path.join(path_output, args.figure_folder)
    tweets_dataset_stem = a2.utils.file_handling.stem_filename(args.filename_tweets)

    ds_raw = utils_scripts.get_dataset(
        args=args, filename=args.filename_tweets, path_figures=path_figures, tracker=tracker
    )

    logger.info(f"Dataset contains {ds_raw.index.shape[0]=} Tweets.")

    ds_exclude_weather_stations = utils_scripts.exclude_and_save_weather_stations_dataset(
        args=args,
        ds_tweets=ds_raw,
        path_output=path_output,
        tweets_dataset_stem=tweets_dataset_stem,
        tracker=tracker,
        path_figures=path_figures,
    )

    indices_train, indices_validate, indices_test = a2.training.training_hugging.split_training_set_tripple(
        ds=ds_exclude_weather_stations,
        key_stratify=args.key_output,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    ds = a2.dataset.load_dataset.reset_index_coordinate(ds_exclude_weather_stations)

    logger.info(
        f"Using {len(indices_validate)} for evaluation, "
        f"{len(indices_test)} for testing and "
        f"{len(indices_train)} for training."
    )

    for suffix, indices in zip(["train", "validate", "test"], [indices_train, indices_validate, indices_test]):
        save_subsection_dataset(
            ds=ds,
            filename=f"{path_output}/{tweets_dataset_stem}_{suffix}.nc",
            indices=indices,
        )
    utils_scripts.plot_and_log_histogram(
        ds, key=args.key_output, path_figures=path_figures, tracker=tracker, filename=f"{args.key_output}_histogram.pdf"
    )
    utils_scripts.plot_and_log_histogram(
        ds,
        key=args.key_precipitation,
        path_figures=path_figures,
        tracker=tracker,
        filename=f"{args.key_precipitation}_histogram.pdf",
    )
    for suffix, indices in zip(["train", "validate", "test"], [indices_train, indices_validate, indices_test]):
        utils_scripts.plot_and_log_histogram(
            ds.sel(index=indices),
            key=args.key_output,
            path_figures=path_figures,
            tracker=tracker,
            filename=f"{args.key_output}_histogram_{suffix}.pdf",
        )
        utils_scripts.plot_and_log_histogram(
            ds.sel(index=indices),
            key=args.key_precipitation,
            path_figures=path_figures,
            tracker=tracker,
            filename=f"{args.key_precipitation}_histogram_{suffix}.pdf",
        )
        utils_scripts.plot_and_log_histogram_2d(
            ds=ds.sel(index=indices),
            x=args.key_output,
            y=args.key_precipitation,
            path_figures=path_figures,
            tracker=tracker,
            annotate=False,
            n_bins=[2, 60],
            ylim=[1e-8, None],
            log=["false", "log"],
            filename=f"{args.key_output}-vs-{args.key_precipitation}_histogram_2d_{suffix}.pdf",
        )


def save_subsection_dataset(ds, filename, indices):
    ds_subsection = ds.sel(index=indices)
    a2.dataset.load_dataset.save_dataset(ds_subsection, filename=filename, reset_index=True, no_conversion=False)


if __name__ == "__main__":
    parser = a2.utils.argparse.get_parser()
    parser.add_argument(
        "--filename_tweets",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/data/tweets/2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc",  # noqa
        help="Filename of training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/data/training_sets_rain_classifier/dataset_split_thresh6M3/",
        help="Path where data saved.",
    )
    parser.add_argument(
        "--figure_folder",
        type=str,
        default="figures/",
        help="Path to saved figures.",
    )
    parser.add_argument(
        "--key_input",
        type=str,
        default="text",
        help="Key that is used for input values.",
    )
    parser.add_argument(
        "--key_output",
        type=str,
        default="raining",
        help="Key that is used for output values.",
    )
    parser.add_argument(
        "--key_precipitation",
        type=str,
        default="tp_h_mm",
        help="Key that specifies precipitation level in dataset.",
    )
    parser.add_argument(
        "--key_precipitation_station",
        type=str,
        default="station_tp_mm",
        help="Key that specifies precipitation level in dataset.",
    )
    parser.add_argument(
        "--key_text",
        type=str,
        default="text_normalized",
        help="Key that specifies texts used in dataset.",
    )
    parser.add_argument(
        "--key_distance_weather_station",
        type=str,
        default="station_distance_km",
        help="Key that specifies distance to nearest weather station.",
    )
    parser.add_argument(
        "--key_raining_station",
        type=str,
        default="raining_station",
        help="Key that specifies presence of rain in weather station dataset.",
    )
    parser.add_argument(
        "--precipitation_threshold_rain_station",
        type=float,
        default=0.1,
        help="Values equal or higher are considered rain.",
    )
    parser.add_argument(
        "--precipitation_threshold_rain",
        type=float,
        default=6e-3,
        help="Values equal or higher are considered rain.",
    )
    parser.add_argument(
        "--maximum_distance_to_station",
        type=float,
        default=1,
        help="Maximum distance to nearest weather station to be considered for weather station dataset.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Percentage of dataset used for validation of model performance.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Percentage of dataset used for testing of model performance.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed to reproduce results.",
    )
    parser.add_argument(
        "--ignore_tracking",
        action="store_true",
        help="Do not use mantik tracking.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to toggle debug mode.",
    )

    args = parser.parse_args()
    main(args)
