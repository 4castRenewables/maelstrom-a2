import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training
import a2.utils
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from a2.training import benchmarks as timer

# from deep500.utils import timer_torch as timer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main(args):
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "maelstrom-a2-eval"
    if args.log_gpu_memory:
        timer.init_cuda()
        timer.reset_cuda_memory_monitoring()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    logging.info(f"Running evaluation as {args.job_id=}")
    logging.info(f"Iteration: {args.iteration=}")
    logging.info(f"Args used: {args.__dict__}")
    tmr = timer.Timer()
    tmr.start(timer.TimeType.IO)
    ds_raw = a2.dataset.load_dataset.load_tweets_dataset(os.path.join(args.data_dir, args.data_filename), raw=True)
    tmr.end(timer.TimeType.IO)
    N_tweets = ds_raw.index.shape[0]
    logging.info(f"loaded {N_tweets} tweets")
    ds_raw["text"] = (["index"], ds_raw[args.key_text].values.copy())
    ds_raw["raining"] = (["index"], np.array(ds_raw[args.key_rain].values > args.threshold_rain, dtype=int))

    indices_train, indices_test = sklearn.model_selection.train_test_split(
        np.arange(ds_raw["index"].shape[0]),
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
        stratify=ds_raw.raining.values,
    )
    logging.info(f"Using {len(indices_test)} for evaluation and {len(indices_train)} for training.")
    model_name = os.path.split(args.model_path)[1]
    logging.info(f"Using ML model: {model_name}")

    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(args.model_path)

    path_figures = os.path.join(args.output_dir, args.figure_folder)
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    tracker.end_run()
    if args.log_gpu_memory:
        timer.reset_cuda_memory_monitoring()
    with tracker.start_run(run_name=args.run_name):
        tracker.log_params(
            {
                "data_description": args.data_description,
                "N_tweets": N_tweets,
                "model_name": model_name,
            }
        )
        tmr.start(timer.TimeType.RUN)
        if args.log_gpu_memory:
            timer.get_cuda_memory_usage("Starting run")
        dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_path)
        tmr.start(timer.TimeType.IO)
        test_ds = dataset_object.build(ds_raw, indices_train, indices_test, train=False)
        tmr.end(timer.TimeType.IO)
        tmr.start(timer.TimeType.EVALUATION)
        trainer = trainer_object.get_trainer(
            test_ds,
            tokenizer=dataset_object.tokenizer,
            disable_tqdm=True,
            callbacks=[
                a2.training.training_deep500.TimerCallback(tmr, gpu=True),
                a2.training.tracking_hugging.LogCallback(tracker),
            ],
            trainer_class=a2.training.training_deep500.TrainerWithTimer,
            evaluate=True,
        )
        tracker.log_params(args.__dict__)
        (
            predictions,
            prediction_probabilities,
        ) = a2.training.evaluate_hugging.predict_dataset(test_ds, trainer)
        if args.log_gpu_memory:
            timer.get_cuda_memory_usage("Finished training")
        tmr.end(timer.TimeType.EVALUATION)
        tmr.end(timer.TimeType.RUN)

        tmr.print_all_time_stats()
        ds_test = a2.training.evaluate_hugging.build_ds_test(
            ds=ds_raw,
            indices_test=indices_test,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )
        truth = ds_test.raining.values

        a2.training.tracking.log_metric_classification_report(tracker, truth, predictions, step=None)

        filename_certainty_plot = os.path.join(path_figures, "plot_2d_predictions_truth.pdf")
        a2.plotting.analysis.plot_prediction_certainty(
            truth=ds_test["raining"].values,
            prediction_probabilities=ds_test["prediction_probability_raining"].values,
            filename=filename_certainty_plot,
        )
        tracker.log_artifact(filename_certainty_plot)

        filename_roc_plot = os.path.join(path_figures, "roc.pdf")
        a2.plotting.analysis.plot_roc(
            ds_test.raining.values, prediction_probabilities[:, 1], filename=filename_roc_plot
        )
        tracker.log_artifact(filename_roc_plot)
        logging.info(f"Max memory consumption [Gbyte]: {timer.get_max_memory_usage()/1e9}")
        if args.log_gpu_memory:
            timer.get_cuda_memory_usage("Finished run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-inpath",
        type=str,
        default=a2.utils.file_handling.get_folder_data() / "tweets/",
        help="Directory where input netCDF-files are stored.",
    )
    parser.add_argument(
        "--data_filename",
        "-infile",
        type=str,
        default="2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered.nc",
        help="Filename of training data.",
    )
    parser.add_argument(
        "--data_description",
        "-datad",
        type=str,
        default="tweets 2017-2020, keywords emojis as description, keywords only",
        help="Data description purely used for logging.",
    )

    parser.add_argument(
        "--model_path",
        "-in",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    parser.add_argument("--run_name", type=str, default="era5 whole dataset", help="Name of run used for logging only.")

    parser.add_argument(
        "--output_dir",
        "-outdir",
        type=str,
        default=a2.utils.file_handling.get_folder_models(),
        help="Directory where output products saved (e.g., figures).",
    )
    parser.add_argument(
        "--figure_folder", type=str, default="figures/", help="Directory where input netCDF-files are stored."
    )

    parser.add_argument("--key_text", type=str, default="text_normalized", help="Key for text in input file.")
    parser.add_argument("--key_rain", type=str, default="tp_h", help="Key for rain data in input file.")
    parser.add_argument(
        "--threshold_rain",
        type=float,
        default=1e-8,
        help="Threshold above which precipitation is considered raining in input file.",
    )

    parser.add_argument("--test_size", "-ts", type=float, default=0.2, help="Fraction of test set.")

    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed value.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")
    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument(
        "--log_gpu_memory",
        action="store_true",
        help="Monitor Cuda memory usage.",
    )
    parser.add_argument(
        "--ignore_tracking",
        action="store_true",
        help="Use mantik tracking.",
    )

    args = parser.parse_args()
    main(args)
