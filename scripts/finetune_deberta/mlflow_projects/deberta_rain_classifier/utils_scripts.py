import logging
import os.path
import pandas as pd

import a2.plotting.analysis
import a2.plotting.histograms
import a2.training.tracking
import a2.utils.file_handling
import numpy as np
from a2.training import benchmarks as timer

FONTSIZE = 16

logger = logging.getLogger(__name__)


def _determine_path_output(args):
    path_output = f"{args.output_dir}/{args.folder_run}/"
    logging.info(f".... using {path_output=}")
    a2.utils.file_handling.make_directories(path_output)
    return path_output


def get_dataset(
    args,
    filename,
    plot_key_distribution=False,
    path_figures=None,
    tracker=None,
    prefix_histogram="",
    setting_rain=True,
):
    ds = a2.dataset.load_dataset.load_tweets_dataset(filename, raw=True, reset_index_raw=True)
    if args.debug:
        logger.info(f"{args.debug=}: Only using 100 first values in dataset!")
        ds = ds.sel(index=slice(0, 100))
    logger.info(f"Input field definition: Setting field {args.key_input=} to {args.key_text=}")
    ds[args.key_input] = (["index"], ds[args.key_text].values.copy())

    N_tweets = ds.index.shape[0]
    logger.info(f"loaded {N_tweets} tweets: from {filename}")

    if setting_rain:
        logger.info(
            f"Output field definition: Setting field {args.key_output=} to "
            f"{args.key_precipitation=} > {args.precipitation_threshold_rain}"
        )
        ds[args.key_output] = (
            ["index"],
            np.array(ds[args.key_precipitation].values > args.precipitation_threshold_rain, dtype=int),
        )
    else:
        ds[args.key_output] = (
            ["index"],
            np.array(ds[args.key_raining].values, dtype=int),
        )
    if plot_key_distribution and path_figures is not None:
        plot_and_log_histogram(
            ds,
            args.key_output,
            path_figures,
            tracker=tracker,
            filename=f"{prefix_histogram}_{args.output}_histogram.pdf",
        )
    return ds


def to_hugging_face_dataset(args, ds, dataset_object):
    dataset = dataset_object.build(
        ds,
        None,
        None,
        key_inputs=args.key_input,
        key_label=args.key_output,
        prediction_dataset=False,
    )
    return dataset


def evaluate_model(
    args,
    ds_test_predicted,
    predictions,
    prediction_probabilities,
    path_figures,
    tracker,
    memory_tracker,
    hyper_parameters=None,
):
    truth = ds_test_predicted[args.key_output].values

    filename_confusion_matrix = os.path.join(path_figures, "confusion_matrix.pdf")
    step = None
    if hyper_parameters is not None:
        step = hyper_parameters.epochs
    a2.training.tracking.log_metric_classification_report(
        tracker=tracker,
        truth=truth,
        predictions=predictions,
        step=step,
        label=args.key_output,
        filename_confusion_matrix=filename_confusion_matrix,
        font_size=FONTSIZE,
    )

    filename_certainty_plot = os.path.join(path_figures, "plot_2d_predictions_truth.pdf")
    a2.plotting.analysis.plot_prediction_certainty(
        truth=ds_test_predicted[args.key_output].values,
        prediction_probabilities=ds_test_predicted[f"prediction_probability_{args.key_output}"].values,
        filename=filename_certainty_plot,
        label_x="True label",
        label_y=f"Prediction probability for '{args.key_output}'",
        font_size=FONTSIZE,
    )
    tracker.log_artifact(filename_certainty_plot)

    filename_roc_plot = os.path.join(path_figures, "roc.pdf")
    a2.plotting.analysis.plot_roc(
        ds_test_predicted[args.key_output].values,
        prediction_probabilities[:, 1],
        filename=filename_roc_plot,
        font_size=FONTSIZE,
    )
    tracker.log_artifact(filename_roc_plot)
    logger.info(f"Max memory consumption [Gbyte]: {timer.get_max_memory_usage()/1e9}")
    if args.log_gpu_memory:
        memory_tracker.get_cuda_memory_usage("Finished run")


def plot_and_log_histogram(ds, key, path_figures, tracker=None, filename="prediction_histogram.pdf"):
    filename_prediction_histogram = os.path.join(path_figures, filename)
    a2.plotting.histograms.plot_histogram(
        key, ds, filename=filename_prediction_histogram, font_size=FONTSIZE, title=key
    )
    if tracker is not None:
        tracker.log_artifact(filename_prediction_histogram)


def plot_and_log_histogram_2d(
    ds,
    x,
    y,
    path_figures,
    n_bins=60,
    tracker=None,
    filename="prediction_histogram.pdf",
    annotate=True,
    **kwargs_histogram,
):
    filename_prediction_histogram = os.path.join(path_figures, filename)
    a2.plotting.histograms.plot_histogram_2d(
        x,
        y,
        df=ds,
        filename=filename_prediction_histogram,
        font_size=FONTSIZE,
        n_bins=n_bins,
        annotate=annotate,
        annotate_color="firebrick",
        **kwargs_histogram,
    )
    if tracker is not None:
        tracker.log_artifact(filename_prediction_histogram)


def exclude_and_save_weather_stations_dataset(
    args,
    ds_tweets,
    path_output,
    tweets_dataset_stem,
    weather_dataset_prefix="Weather_stations_",
    tracker=None,
    path_figures=None,
):
    ds_weather_stations = ds_tweets.where(
        ds_tweets[args.key_distance_weather_station] <= args.maximum_distance_to_station, drop=True
    )

    ds_weather_stations[args.key_raining_station] = (
        ["index"],
        np.array(
            ds_weather_stations[args.key_precipitation_station].values > args.precipitation_threshold_rain_station,
            dtype=int,
        ),
    )

    try:
        plot_and_log_histogram(
            ds_weather_stations,
            key=args.key_raining_station,
            path_figures=path_figures,
            tracker=tracker,
            filename=f"weather_stations_{args.key_raining_station}_histogram.pdf",
        )
        plot_and_log_histogram(
            ds_weather_stations,
            key=args.key_precipitation_station,
            path_figures=path_figures,
            tracker=tracker,
            filename=f"weather_stations_{args.key_precipitation_station}_histogram.pdf",
        )
        plot_and_log_histogram_2d(
            ds_weather_stations,
            x=args.key_raining_station,
            y=args.key_output,
            n_bins=2,
            path_figures=path_figures,
            tracker=tracker,
            filename=f"weather_stations_{args.key_raining_station}-{args.key_output}_histogram2d.pdf",
        )
    except Exception as e:
        raise ValueError("Cannot save plots for weather station dataset!") from e
    a2.dataset.load_dataset.save_dataset(
        ds_weather_stations,
        filename=f"{path_output}{weather_dataset_prefix}{tweets_dataset_stem}.nc",
    )
    # https://stackoverflow.com/questions/52417929/remove-elements-from-one-array-if-present-in-another-array-keep-duplicates-nu
    all_indices = ds_tweets.index.values
    weather_station_indices = ds_weather_stations.index.values
    remaining_indices = all_indices[np.isin(all_indices, weather_station_indices, invert=True)]
    ds_tweets_keywords_near_stations_excluded = ds_tweets.sel(index=remaining_indices)
    logger.info(
        f"Weather station dataset contains {len(weather_station_indices)} Tweets. "
        f"Remaining indices are {len(remaining_indices)} Tweets."
    )
    return ds_tweets_keywords_near_stations_excluded


def load_power(filename):
    power = pd.read_csv(filename).drop(columns=["Unnamed: 0"])
    power = power.set_index("timestamps")
    power.index = power.index - min(power.index)
    return power


def total_power_consumption_Wh_per_device(power):
    return np.sum(power[power.columns].values[1:] * np.diff(power.index.values)[:, None] / 3600, axis=0)


def average_consumption_W_per_device(power):
    return np.mean(power[power.columns].values[1:], axis=0)


def total_power_consumption_Wh(power):
    return sum(total_power_consumption_Wh_per_device(power))
