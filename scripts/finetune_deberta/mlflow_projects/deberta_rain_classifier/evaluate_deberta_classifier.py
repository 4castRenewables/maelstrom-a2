import argparse
import logging
import os

import a2.dataset
import a2.plotting
import a2.training.benchmarks
import a2.training.model_configs
import a2.training.tracking
import a2.training.tracking_hugging
import a2.training.training_deep500
import a2.training.training_hugging
import a2.utils.argparse
import utils_scripts
from a2.training import benchmarks as timer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main(args):
    tracker = a2.training.tracking.Tracker(ignore=args.ignore_tracking)
    memory_tracker = a2.training.benchmarks.CudaMemoryMonitor()
    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
    if args.debug:
        logger.info("RUNNING IN DEBUG MODE!")
    logger.info(f"Running finetuning as {args.job_id=}")
    logger.info(f"Iteration: {args.iteration=}")
    logger.info(f"Args used: {args.__dict__}")
    logger.info(f"Evaluating saved model {args.model_trained_path=}")

    dataset_object = a2.training.dataset_hugging.DatasetHuggingFace(args.model_trained_path)

    path_output = utils_scripts._determine_path_output(args)
    path_figures = os.path.join(path_output, args.folder_figures)

    ds_evaluate = utils_scripts.get_dataset(
        args,
        f"{args.filename_dataset_evaluate}",
        path_figures=path_figures,
        tracker=tracker,
        prefix_histogram="evaluation",
        setting_rain=False,  # already done in build_dataset_rain_classifier.py
    )
    dataset_evaluate = utils_scripts.to_hugging_face_dataset(args=args, ds=ds_evaluate, dataset_object=dataset_object)

    trainer_object = a2.training.training_hugging.HuggingFaceTrainerClass(
        args.model_trained_path, num_labels=args.num_labels
    )

    if args.log_gpu_memory:
        memory_tracker.reset_cuda_memory_monitoring()

    tmr = timer.Timer()
    tracker.log_params(
        {
            "N_tweets_evaluate": len(dataset_evaluate["index"]),
        }
    )
    tmr.start(timer.TimeType.RUN)
    if args.log_gpu_memory:
        memory_tracker.get_cuda_memory_usage("Starting run")
    trainer_class = a2.training.model_configs.get_customized_trainer_class(
        args.trainer_name,
        method_overwrites=[
            args.loss,
        ],
    )
    trainer = trainer_object.get_trainer(
        dataset=None,
        tokenizer=dataset_object.tokenizer,
        fp16=True if a2.training.utils_training.gpu_available() else False,
        trainer_class=trainer_class,
        label=args.key_output,
        evaluate=True,
    )
    tmr.start(timer.TimeType.EVALUATION)
    (
        predictions,
        prediction_probabilities,
    ) = a2.training.evaluate_hugging.predict_dataset(dataset_evaluate, trainer)
    if args.log_gpu_memory:
        memory_tracker.get_cuda_memory_usage("Finished training")
    tmr.end(timer.TimeType.EVALUATION)
    tmr.end(timer.TimeType.RUN)

    tmr.print_all_time_stats()
    ds_evaluate_predicted = a2.training.evaluate_hugging.build_ds_test(
        ds=ds_evaluate,
        indices_test=None,
        predictions=predictions,
        prediction_probabilities=prediction_probabilities,
        label=args.key_output,
    )
    utils_scripts.evaluate_model(
        args,
        ds_evaluate_predicted,
        predictions,
        prediction_probabilities,
        path_figures,
        tracker,
        memory_tracker,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument(
        "--filename_dataset_evaluate", required=True, type=str, help="Filename of dataset used for evaluation."
    )
    # /p/project/deepacf/maelstrom/ehlert1/data/training_sets_rain_classifier/dataset_split_thresh6M3/Weather_stations_2017_2020_tweets_rain_sun_vocab_emojis_locations_bba_Tp_era5_no_bots_normalized_filtered_weather_stations_fix_predicted_simpledeberta_radar.nc
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
        "--key_raining",
        type=str,
        default="raining_station",
        help="Key that specifies labels used for raining.",
    )
    parser.add_argument(
        "--key_text",
        type=str,
        default="text_normalized",
        help="Key that specifies texts used in dataset.",
    )

    # MODEL
    parser.add_argument(
        "--model_trained_path",
        "-in",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/models/trained_models/rain_classification/dataset_split_thresh6M3/checkpoint-22209/",  # noqa: E501
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
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels in classification task.",
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

    # LOGS
    parser.add_argument(
        "--log_gpu_memory",
        action="store_true",
        help="Monitor Cuda memory usage.",
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

    # OUTPUTS
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/p/project/deepacf/maelstrom/ehlert1/data/training_sets_rain_classifier/",
        help="Path to model.",
    )
    parser.add_argument(
        "--folder_run",
        type=str,
        default="dataset_split_thresh6M3/",
        help="Relative path to all ouputs connected to this run.",
    )
    parser.add_argument(
        "--folder_figures",
        type=str,
        default="figures/",
        help="Relative path to `output_dir` for saving figures.",
    )

    parser.add_argument("--job_id", "-jid", type=int, default=None, help="Job id when running on hpc.")
    parser.add_argument("--iteration", "-i", type=int, default=0, help="Iteration number when running benchmarks.")
    args = parser.parse_args()
    main(args)
