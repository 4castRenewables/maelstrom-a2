import logging
import os.path

import a2.plotting.analysis
import a2.plotting.histograms
import a2.training.tracking
import a2.utils.file_handling
import numpy as np
from a2.training import benchmarks as timer

FONTSIZE = 16


def _determine_path_output(args):
    path_output = f"{args.output_dir}/{args.task_name}/"
    logging.info(f".... using {path_output=}")
    a2.utils.file_handling.make_directories(path_output)
    return path_output


def get_dataset(
    args,
    filename,
    dataset_object,
    set_labels=False,
    build_hugging_dataset=True,
    plot_key_distribution=False,
    path_figures=None,
    tracker=None,
    prefix_histogram="",
):
    ds_raw = a2.dataset.load_dataset.load_tweets_dataset(filename, raw=True)
    if args.debug:
        ds_raw = ds_raw.sel(index=slice(0, 100))
    ds_raw[args.key_input] = (["index"], ds_raw[args.key_text].values.copy())

    N_tweets = ds_raw.index.shape[0]
    logging.info(f"loaded {N_tweets} tweets: from {filename}")
    if set_labels:
        if args.classifier_domain == "rain":
            ds_raw[args.key_output] = (
                ["index"],
                np.array(ds_raw[args.key_rain].values > args.threshold_rain, dtype=int),
            )
        elif args.classifier_domain == "relevance":
            ds_raw[args.key_output] = (["index"], np.array(ds_raw[args.key_relevance].values, dtype=int))
        else:
            raise NotImplementedError(f"{args.classifier_domain} not implemented!")
    dataset = None
    if build_hugging_dataset:
        dataset = dataset_object.build(
            ds_raw,
            None,
            None,
            key_inputs=args.key_input,
            key_label=args.key_output,
            prediction_dataset=not set_labels,
        )
    if plot_key_distribution and path_figures is not None:
        plot_and_log_histogram(
            ds_raw,
            args.key_output,
            path_figures,
            tracker=tracker,
            filename=f"{prefix_histogram}_{args.output}_histogram.pdf",
        )
    return dataset, ds_raw


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
    logging.info(f"Max memory consumption [Gbyte]: {timer.get_max_memory_usage()/1e9}")
    if args.log_gpu_memory:
        memory_tracker.get_cuda_memory_usage("Finished run")


def plot_and_log_histogram(ds, key, path_figures, tracker=None, filename="prediction_histogram.pdf"):
    filename_prediction_histogram = os.path.join(path_figures, filename)
    a2.plotting.histograms.plot_histogram(
        key, ds, filename=filename_prediction_histogram, font_size=FONTSIZE, title=key
    )
    if tracker is not None:
        tracker.log_artifact(filename_prediction_histogram)


def exclude_and_save_weather_stations_dataset(
    args, ds_tweets, path_output, filename_prefix, tracker=None, path_figures=None
):
    ds_weather_stations = ds_tweets.where(
        ds_tweets[args.key_distance_weather_station] <= args.kms_within_station, drop=True
    )
    try:
        plot_and_log_histogram(
            ds_weather_stations,
            key="raining_station",
            path_figures=path_figures,
            tracker=tracker,
            filename="weather_stations_raining_histogram.pdf",
        )
        plot_and_log_histogram(
            ds_weather_stations,
            key="station_tp_mm",
            path_figures=path_figures,
            tracker=tracker,
            filename="weather_stations_precipitation_histogram.pdf",
        )
    except Exception as e:
        raise ValueError("Cannot save plots for weather station dataset!") from e
    a2.dataset.load_dataset.save_dataset(
        ds_weather_stations,
        filename=f"{path_output}{args.weather_station_dataset_prefix}{filename_prefix}.nc",
    )
    # https://stackoverflow.com/questions/52417929/remove-elements-from-one-array-if-present-in-another-array-keep-duplicates-nu
    all_indices = ds_tweets.index.values
    weather_station_indices = ds_weather_stations.index.values
    print(f"{all_indices=}")
    print(f"{weather_station_indices=}")
    remaining_indices = all_indices[np.isin(all_indices, weather_station_indices, invert=True)]
    print(f"{remaining_indices=}")
    ds_tweets_keywords_near_stations_excluded = ds_tweets.sel(index=remaining_indices)
    return ds_tweets_keywords_near_stations_excluded
